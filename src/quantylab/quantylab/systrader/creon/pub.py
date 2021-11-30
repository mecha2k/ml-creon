import argparse
import asyncio
import json
import logging

from quantylab.systrader.creon import Creon
from quantylab.common.db import redis
from quantylab.common import util


logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s %(funcName)s %(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--connect', action='store_true', default=False)
    parser.add_argument('--id')
    parser.add_argument('--pwd')
    parser.add_argument('--pwdcert')
    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    c = Creon()
    # if not c.connected():
    #     if args.connect:
    #         c.connect(args.id, args.pwd, args.pwdcert)
    #     else:
    #         sys.exit(1)

    r = redis.get_client()
    def cb(item):
        item['date'] = util.get_str_today()
        m = json.dumps(item, ensure_ascii=False)
        logger.debug('[stockcur] {}'.format(m))
        # r.publish('stockcur', m)
        r.set(item['code'], m)
    stockranks = json.loads(r.get('stockranks'))
    stockranks = stockranks['stockranks']
    codes = [item['code'] for item in stockranks]
    for code in codes:
        c.subscribe_stockcur(code, cb)
    
    try:
        loop.run_forever()
    except KeyboardInterrupt as e:
        logger.debug('Unsubscribing...')
        c.unsubscribe_stockcur()
