import bz2
import os
import shutil
import sys
import json
import tempfile
import subprocess
import time

import gensim
import numpy as np


class TimeChecker(object):

    def __enter__(self):
        self.start_times = os.times()
        return self

    def __exit__(self, type, value, traceback):
        pass

    def status(self):
        current_times = os.times()
        fields = ('utime', 'stime', 'cutime', 'cstime', 'elapsed_time')
        return {
            field: current_value - start_value
            for field, (start_value, current_value) in zip(fields, zip(self.start_times, current_times))
        }


def infer_stochastic_matrix(alphas, a=0, tol=1e-10):
    """
    Infer stochastic matrix from Dirichlet distribution.

    Input: matrix with rows corresponding to parameters of
    the asymmetric Dirichlet distributions, parameter a.

    a=0 => expected distributions
    a=1 => most probable distributions
    a=1/2 => normalized median-marginal distributions

    Returns: inferred stochastic matrix.
    """
    assert isinstance(alphas, np.ndarray)
    alpha0 = alphas.sum(axis=1, keepdims=True)
    A = alphas - a
    A[A < tol] = 0
    A /= A.sum(axis=1, keepdims=True) + 1e-15
    return A


def compute_perplexity_artm(corpus, Phi, Theta):
    sum_n = 0.0
    sum_loglike = 0.0
    for doc_id, doc in enumerate(corpus):
        for term_id, count in doc:
            sum_n += count
            sum_loglike += count * np.log( np.dot(Theta[doc_id, :], Phi[:, term_id]) )
    perplexity = np.exp(- sum_loglike / sum_n)
    return perplexity


### Interface for OnlineLDA implementation

def run_impl(impl, name, train, test, wordids,
             num_processors=1, num_topics=100, batch_size=10000, passes=1,
             kappa=0.5, tau0=64, alpha=0.1, beta=0.1, num_inner_iters=20):
    """
    Run OnlineLDA algorithm.

    :param name: name of experiment
    :param train: train corpus
    :param test: dict {id: <test corpus>} where id is 'test' and 'valid'
    :param wordids: id-to-word mapping file (gensim Dictionary)
    :param num_processors: number of processors to use (more than one means parallelization)
    :param num_topics: number of topics
    :param batch_size: number of documents in the batch
    :param passes: number of passes through the train corpus
    :param kappa: power of learning rate
    :param tau0: initial learning rate
    :param alpha: document distribution smoothing coefficient (parameter dirichlet prior)
    :param beta: topic distribution smoothing coefficient (parameter dirichlet prior)
    :param num_inner_iters: maximal number of inner iterations on E-step
    """
    func = globals()['run_%s' % impl]
    func(name, train, test, wordids,
         num_processors=num_processors, num_topics=num_topics, batch_size=batch_size, passes=passes,
         kappa=kappa, tau0=tau0, alpha=alpha, beta=beta, num_inner_iters=num_inner_iters)


### Gensim
#
# Website: http://radimrehurek.com/gensim/
# Tutorial: http://radimrehurek.com/gensim/tut2.html
#

def run_gensim(name, train, test, wordids,
               num_processors=1, num_topics=100, batch_size=10000, passes=1,
               kappa=0.5, tau0=1.0, alpha=0.1, beta=0.1, update_every=1, num_inner_iters=20):

    if tau0 != 1.0:
        raise ValueError('Gensim does not support tau0 != 1.0')

    id2word = gensim.corpora.Dictionary.load_from_text('data/%s' % wordids)
    train_corpus = gensim.corpora.MmCorpus('data/%s.mm' % train)

    gamma_threshold = 0.001

    with TimeChecker() as timer:
        if num_processors == 1:
            model = gensim.models.LdaModel(
                corpus=train_corpus,
                id2word=id2word,
                num_topics=num_topics,
                distributed=False,
                chunksize=batch_size,
                passes=passes,
                update_every=update_every,
                alpha=alpha,
                eta=beta,
                decay=kappa,
                eval_every=None,
                iterations=num_inner_iters,
                gamma_threshold=gamma_threshold,
            )
        else:
            model = gensim.models.LdaMulticore(
                corpus=train_corpus,
                id2word=id2word,
                num_topics=num_topics,
                chunksize=batch_size,
                passes=passes,
                batch=False,
                alpha=alpha,
                eta=beta,
                decay=kappa,
                eval_every=None,
                iterations=num_inner_iters,
                gamma_threshold=gamma_threshold,
                workers=num_processors-1, # minus one because `workers`
                                          # is the number of extra processes
            )
        train_time = timer.status()

    model.save('target/%s.gensim_model' % name)

    report = {
        'train_time': train_time,
    }

    Lambda = model.state.get_lambda()
    Phi = infer_stochastic_matrix(Lambda, 0)
    matrices = {
        'Lambda': Lambda,
        'Phi_mean': Phi,
        'Phi_map': infer_stochastic_matrix(Lambda, 1),
    }

    for id, corpus_name in test.iteritems():
        test_corpus = gensim.corpora.MmCorpus('data/%s.mm' % corpus_name)

        with TimeChecker() as timer:
            Gamma, _ = model.inference(test_corpus)
            inference_time = timer.status()

        Theta = infer_stochastic_matrix(Gamma, 0)
        matrices['%s_Gamma' % id] = Gamma
        matrices['%s_Theta_mean' % id] = Theta
        matrices['%s_Theta_map' % id] = infer_stochastic_matrix(Gamma, 1)

        report[id] = {
            'inference_time': inference_time,
            'perplexity_gensim': np.exp(-model.log_perplexity(test_corpus)),
            'perplexity_artm': compute_perplexity_artm(test_corpus, Phi, Theta),
        }

    with open('target/%s.report.json' % name, 'w') as report_file:
        json.dump(report, report_file, indent=2)
    np.savez_compressed('target/%s.matrices.npz' % name, **matrices)


### Vowpal Wabbit
#
# Website: https://github.com/JohnLangford/vowpal_wabbit/wiki
# LDA Tutorial: https://github.com/JohnLangford/vowpal_wabbit/wiki/Latent-Dirichlet-Allocation
#

def run_vw(name, train, test, wordids,
           num_processors=1, num_topics=100, batch_size=10000, passes=1,
           kappa=0.5, tau0=64, alpha=0.1, beta=0.1, update_every=1, num_inner_iters=20, limit_docs=None):

    def read_vw_matrix(filename, topics=False, n_term=None):
        with open(filename) as f:
            if topics:
                for i in xrange(11): f.readline()
            result_matrix = []
            for line in f:
                parts = line.strip().replace('  ', ' ').split(' ')
                if topics:
                    index = int(parts[0])
                    matrix_line = map(float, parts[1:])
                    if index < n_term or not n_term:
                        result_matrix.append(matrix_line)
                else:
                    index = int(parts[-1])
                    matrix_line = map(float, parts[:-1])
                    result_matrix.append(matrix_line)
        return np.array(result_matrix, dtype=float)

    def read_vw_gammas(predictions_path):
        """
        Read matrix of inferred document distributions (gammas) from vw predictions file.
        :return: np.ndarray, size = num_docs x num_topics
        """
        gammas = read_vw_matrix(predictions_path, topics=False)
        return gammas

    def read_vw_lambdas(topics_path, n_term=None):
        """
        Read matrix of inferred topic distributions (lambdas) from vw readable model file.
        :param n_term: number of words
        :return: np.ndarray, size = num_topics x num_terms
        """
        lambdas = read_vw_matrix(topics_path, topics=True, n_term=n_term).T
        return lambdas

    if num_processors != 1:
        raise ValueError('Vowpal Wabbit LDA does not support parallelization')
    if update_every != 1:
        raise ValueError('Vowpal Wabbit LDA does not support update_every != 1')

    id2word = gensim.corpora.Dictionary.load_from_text('data/%s' % wordids)
    train_corpus = gensim.corpora.MmCorpus('data/%s.mm' % train)

    tempdir = tempfile.mkdtemp()

    cmd = [
        'vw',
        'data/%s.vw' % train,
        '-b', '%.0f' % np.ceil(np.log2(len(id2word))),
        '--cache_file', os.path.join(tempdir, 'cache_file'),
        '--random_seed', str(123),
        '--lda', str(num_topics),
        '--lda_alpha', str(alpha),
        '--lda_rho', str(beta),
        '--lda_D', str(train_corpus.num_docs),
        '--minibatch', str(batch_size),
        '--power_t', str(kappa),
        '--initial_t', str(tau0),
        '--passes', str(passes),
        '--readable_model', os.path.join(tempdir, 'readable_model'),
        '-p', os.path.join(tempdir, 'predictions'),
        '-f', 'target/%s.vw_model' % name,
    ]
    if limit_docs:
        cmd += ['--examples', str(limit_docs)]

    with TimeChecker() as timer:
        proc = subprocess.Popen(cmd)
        proc.wait()
        train_time = timer.status()
    shutil.rmtree(tempdir)

    report = {
        'train_time': train_time,
    }

    Lambda = read_vw_lambdas(os.path.join(tempdir, 'readable_model'), n_term=len(id2word))
    Phi = infer_stochastic_matrix(Lambda, 0)
    matrices = {
        'Lambda': Lambda,
        'Phi_mean': Phi,
        'Phi_map': infer_stochastic_matrix(Lambda, 1),
    }

    for id, corpus_name in test.iteritems():
        test_corpus = gensim.corpora.MmCorpus('data/%s.mm' % corpus_name)

        predictions_path = os.path.join(tempdir, 'predictions_%s' % id)
        cmd = [
            'vw',
            'data/%s.vw' % corpus_name,
            '--minibatch', str(test_corpus.num_docs),
            '--initial_regressor', 'target/%s.vw_model' % name,
            '-p', predictions_path,
        ]

        with TimeChecker() as timer:
            proc = subprocess.Popen(cmd)
            proc.wait()
            inference_time = timer.status()

        Gamma = read_vw_gammas(predictions_path)
        Theta = infer_stochastic_matrix(Gamma, 0)
        matrices['%s_Gamma' % id] = Gamma
        matrices['%s_Theta_mean' % id] = Theta
        matrices['%s_Theta_map' % id] = infer_stochastic_matrix(Gamma, 1)

        report[id] = {
            'inference_time': inference_time,
            'perplexity_artm': compute_perplexity_artm(test_corpus, Phi, Theta),
        }

    with open('target/%s.report.json' % name, 'w') as report_file:
        json.dump(report, report_file, indent=2)
    np.savez_compressed('target/%s.matrices.npz' % name, **matrices)



### BigARTM
#
# Website: http://bigartm.org/
#

def run_bigartm(name, train, test, wordids,
                num_processors=1, num_topics=100, batch_size=10000, passes=1,
                kappa=0.5, tau0=64, alpha=0.1, beta=0.1, update_every=1, num_inner_iters=20):

    import artm.messages_pb2, artm.library

    # TODO: generate chunk files
    train_batches_folder = 'data/wiki_bow_test_batch_10k/'
    test_batches_folder = 'data/wiki_bow_test_batch_10k/'

    unique_tokens = artm.library.Library().LoadDictionary(train_batches_folder + 'dictionary')

    master_config = artm.messages_pb2.MasterComponentConfig()
    master_config.processors_count = num_processors
    master_config.cache_theta = True
    master_config.disk_path = train_batches_folder

    perplexity_collection_config = artm.messages_pb2.PerplexityScoreConfig()
    perplexity_collection_config.model_type = artm.library.PerplexityScoreConfig_Type_UnigramCollectionModel
    perplexity_collection_config.dictionary_name = unique_tokens.name


    with artm.library.MasterComponent(master_config) as master:
        dictionary = master.CreateDictionary(unique_tokens)
        perplexity_score = master.CreatePerplexityScore(config = perplexity_collection_config)
        smooth_sparse_phi = master.CreateSmoothSparsePhiRegularizer()
        smooth_sparse_theta = master.CreateSmoothSparseThetaRegularizer()

        items_processed_score = master.CreateItemsProcessedScore()

        # Configure the model
        model = master.CreateModel(
            config=artm.messages_pb2.ModelConfig(),
            topics_count=num_topics,
            inner_iterations_count=num_inner_iters
        )
        model.EnableScore(items_processed_score)
        model.EnableRegularizer(smooth_sparse_phi, beta)
        model.EnableRegularizer(smooth_sparse_theta, alpha)

        model.Initialize(dictionary)

        with TimeChecker() as timer:

            master.InvokeIteration(passes)
            done = False
            first_sync = True
            next_items_processed = (batch_size * update_every)
            while (not done):
                done = master.WaitIdle(10)    # Wait 10 ms and check if the number of processed items had changed
                current_items_processed = items_processed_score.GetValue(model).value
                if done or (current_items_processed >= next_items_processed):
                    update_count = current_items_processed / (batch_size * update_every)
                    next_items_processed = current_items_processed + (batch_size * update_every)      # set next model update
                    rho = pow(tau0 + update_count, -kappa)                                            # calculate rho
                    model.Synchronize(decay_weight=(0 if first_sync else (1-rho)), apply_weight=rho)  # synchronize model
                    first_sync = False
                    print "Items processed: %i, Elapsed time: %.3f " % (
                        timer.status()['elapsed_time'], current_items_processed)

            report = {'train_time': timer.status()}
            with open('target/%s.report.json' % name, 'w') as report_file:
                json.dump(report, report_file)

        print "Saving topic model... ",
        with open('target/%s.bigartm_model' % name, 'wb') as binary_file:
            binary_file.write(master.GetTopicModel(model).SerializeToString())
        print "Done. "


