import os
import logging
import gensim
import bz2
import json
import argparse

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

id2word = gensim.corpora.Dictionary.load_from_text('wiki_corpus/enwiki-20141208-pages-articles_wordids.txt.bz2')
mm = gensim.corpora.MmCorpus('wiki_corpus/enwiki-20141208-pages-articles_tfidf.mm')


def experiment_speed_lda(num_topics=100, update_every=1, chunksize=10000, passes=1):
	name = 'gensim_lda_topics%d_updateevery%d_chunksize%d_passes%d' % (
			num_topics, update_every, chunksize, passes,
		)

	times_start = os.times()
	model = gensim.models.ldamodel.LdaModel(
		corpus=mm,
		id2word=id2word,
		num_topics=num_topics,
		update_every=update_every,
		chunksize=chunksize,
		passes=passes,
	)
	times_finish = os.times()

	model.save('ex_gensim/' + name)

	with open('ex_gensim/' + name + '.times', 'w') as fout:
		report = {
			'times': {'start': times_start, 'finish': times_finish},
			'model': {
				'class': 'gensim.models.ldamodel.LdaModel',
				'num_topics': num_topics,
				'update_every': update_every,
				'chunksize': chunksize,
				'passes': passes,
			},
			'elapsed_time': {
				'user': times_finish[0] - times_start[0],
				'system': times_finish[1] - times_start[1],
				'children_user': times_finish[2] - times_start[2],
				'children_system': times_finish[3] - times_start[3],
				'real': times_finish[4] - times_start[4],
			},
		}
		fout.write(json.dumps(report, indent=2))


if __name__ == '__main__':

	# Issue about measuring time:
	# http://stackoverflow.com/questions/7421641/measuring-elapsed-time-in-python

	parser = argparse.ArgumentParser(description='Speed experiment with gensim')
	parser.add_argument('--num_topics', type=int, default=100)
	parser.add_argument('--update_every', type=int, default=1)
	parser.add_argument('--chunksize', type=int, default=10000)
	parser.add_argument('--passes', type=int, default=1)
	args = parser.parse_args()

	experiment_speed_lda(num_topics=args.num_topics, update_every=args.update_every, 
		chunksize=args.chunksize, passes=args.passes)

