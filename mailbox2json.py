#!/usr/bin/env python

import tarfile
import collections
import time
import email
import argparse
import operator
import logging
import json

import bs4
import dateutil
import pyzmail


logger = logging.getLogger(__name__)


def html_to_plain(html):
    bs = bs4.BeautifulSoup(html, 'lxml')
    return bs.text


def extract_message_text(msg):
    text_plain = None
    text_type = None

    body_part = msg.text_part or msg.html_part
    if body_part is not None:
        body_content, body_charset = pyzmail.decode_text(
            body_part.get_payload(),
            body_part.charset,
            None,
        )
        text_type = body_part.type
        if body_part.type == 'text/html':
            text_plain = html_to_plain(body_content)
        else:
            text_plain = body_content
    
    return text_type, text_plain


def address_to_obj((name, email_addr)):
    return {
        'name': name,
        'email': email_addr,
    }


def parse_message(msg):
    timestamp = time.mktime(email.utils.parsedate(msg.get_decoded_header('date')))
    #timestamp_iso = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%dT%H:%M:%S')

    body_type, body_text = extract_message_text(msg)

    attachments = [
        {
            'filename': mailpart.filename,
            'type': mailpart.type,
            'size': len(mailpart.get_payload()),
        }
        for mailpart in msg.mailparts
        if not mailpart.is_body
    ]

    message_record = {
        'timestamp': timestamp,
        'subject': msg.get_subject(),
        'body_text': body_text,
        'body_type': body_type,
        'attachments': attachments,
        'addresses': {
            'from': address_to_obj(msg.get_address('from')),
            'to': map(address_to_obj, msg.get_addresses('to')),
            'cc': map(address_to_obj, msg.get_addresses('cc')),
            'bcc': map(address_to_obj, msg.get_addresses('bcc')),
        },
        'message_id': msg.get_decoded_header('message-id', default=None),
        'in_reply_to': msg.get_decoded_header('in-reply-to', default=None),
        'delivered_to': msg.get_decoded_header('delivered-to', default=None),
    }
    
    return message_record


def check_mime(msg):
    return any(msg.has_key(key) for key in ('mime-version', 'message-id', 'delivered-to')) 

def mailbox_to_json(archive, output_file):
    message_member = {}
    message_thread = {}
    message_info = {}

    def merge_threads(msg_id, related_ids):
        messages_to_update = set()
        threads_to_update = set()

        new_thread_id = message_thread[msg_id]
        for rel_id in related_ids:
            if rel_id in message_thread:
                new_thread_id = min(new_thread_id, message_thread[rel_id])
                threads_to_update.add(message_thread[rel_id])
            else:
                messages_to_update.add(rel_id)

        # update messages
        for rel_id in messages_to_update:
            message_thread[rel_id] = new_thread_id

        # merge threads
        for rel_id, rel_thread_id in message_thread.iteritems():
            if rel_thread_id in threads_to_update:
                message_thread[rel_id] = new_thread_id

    n_no_message_id = 0

    logger.info('Start indexing mail headers')
    
    for n, member in enumerate(archive):
        if member.isfile():
            f = archive.extractfile(member)
            msg = pyzmail.parse.PyzMessage.factory(f)

            if not check_mime(msg):
                logger.warning('File {name} is not a MIME message'.format(name=member.name))
                continue

            msg_id = msg.get_decoded_header('message-id')
            msg_reply_to = msg.get_decoded_header('in-reply-to').split()
            msg_references = msg.get_decoded_header('references').split()

            if msg_id is None or len(msg_id) == 0:
                msg_id = 'NO_MESSAGE_ID_%.10d' % n_no_message_id
                logger.warning('Message {name} has no Message-ID, assign "{new_id}"'.format(
                        name=member.name, new_id=msg_id))
                n_no_message_id += 1

            if msg_id in message_member:               
                for suffix in xrange(100500):
                    new_msg_id = msg_id + ('_DUPLICATE_%d' % suffix)
                    if new_msg_id not in message_member:
                        break
                        
                logger.warning('Duplicated Message-ID "{message_id}" in {name2} (previously occured in {name1}),'
                                ' replaced to "{new_message_id}"'.format(
                        message_id=msg_id,
                        new_message_id=new_msg_id,
                        name1=message_member[msg_id].name,
                        name2=member.name,
                    ))
                
                msg_id = new_msg_id

            next_thread_id = n
            message_thread[msg_id] = next_thread_id
            merge_threads(msg_id, set(msg_references + msg_reply_to))

            message_member[msg_id] = member            
            message_info[msg_id] = {
                'subject': msg.get_subject(),
                'timestamp': time.mktime(email.utils.parsedate(msg.get_decoded_header('date'))),
            }

        if n % 1000 == 0 and n > 0:
            logger.info('Indexed {n} files, found {n_messages} messages'.format(
                    n=n,
                    n_messages=len(message_member),
                ))
    
    logging.info('Collecting mail threads')
    
    thread_messages = {}
    thread_start_timestamp = {}
    
    for msg_id, thread_id in message_thread.iteritems():
        if thread_id not in thread_messages:
            thread_messages[thread_id] = []
            thread_start_timestamp[thread_id] = None
        thread_messages[thread_id].append(msg_id)
        if msg_id in message_info:
            ts = message_info[msg_id]['timestamp']
            if thread_start_timestamp[thread_id] is None or thread_start_timestamp[thread_id] > ts:
                thread_start_timestamp[thread_id] = ts

    logging.info('Messages: {n_messages}, Threads: {n_threads}'.format(
            n_messages=len(message_member),
            n_threads=len(thread_messages),
        ))
                
    logging.info('Start writing JSON dataset')
    
    threads_sorted_by_time = sorted(thread_messages, key=thread_start_timestamp.get)
    for n, thread_id in enumerate(threads_sorted_by_time):
        messages = []
        for msg_id in thread_messages[thread_id]:
            if msg_id in message_member:
                msg_file = archive.extractfile(message_member[msg_id])
                msg = pyzmail.parse.PyzMessage.factory(msg_file)
                msg_record = parse_message(msg)
                messages.append(msg_record)

        messages.sort(key=operator.itemgetter('timestamp'))
        
        if len(messages) == 0:
            continue

        participants = {}
        for msg in messages:
            msg_addresses = [msg['addresses']['from']] + [
                entry 
                for lst in ('to', 'cc', 'bcc') 
                for entry in msg['addresses'][lst]
            ]
            for entry in msg_addresses:
                new_name = entry['name']
                old_name = participants.get(entry['email'])
                if old_name is None or len(old_name) < new_name or old_name.lower() == entry['email'].lower():
                    participants[entry['email']] = new_name

        thread_record = {
            'start_timestamp': min([msg['timestamp'] for msg in messages]),
            'last_timestamp': max([msg['timestamp'] for msg in messages]),
            'messages': messages,
            'subject': messages[0]['subject'],
            'participants': [
                {'name': name, 'email': email_addr}
                for email_addr, name in participants.iteritems()
            ]
        }

        output_file.write(
            json.dumps(thread_record, separators=(',', ':')) + '\n'
        )

        if n % 1000 == 0 and n > 0:
            logger.info('Writed {n} threads'.format(n=n))


if __name__ == '__main__':
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG,
    )
    
    parser = argparse.ArgumentParser(description='Convert mailbox to JSON dataset')
    parser.add_argument('mailbox_archive')
    parser.add_argument('jsonlines_output')
    args = parser.parse_args()
    
    archive = tarfile.open(args.mailbox_archive)
    with open(args.jsonlines_output, 'w') as output_file:
        mailbox_to_json(archive, output_file)

        
