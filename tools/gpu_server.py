import pika
import time
import math
import threading
import functools
import termcolor
import datetime
import pika
import functools
import pickle
import os
import math
import torch
from test_supernet import evaluate_operations
from multiprocessing.pool import ThreadPool
print=functools.partial(print,flush=True)
tostring=lambda *args:' '.join(map(str,args))
printred=lambda *args,**kwargs:termcolor.cprint(tostring(*args),color='red',flush=True,**kwargs)
printgreen=lambda *args,**kwargs:termcolor.cprint(tostring(*args),color='green',flush=True,**kwargs)
def info_prefix():
    return '[{} info]'.format(datetime.datetime(1,1,1).now())


class GpuServer:

    def __init__(self, ):
        self._task_queue_name = 'task'
        self._result_queue_name = 'result'
        self._host = 'localhost'
        self._port = '23333'
        self._thread = None
        self._usrname = 'test'
        self._heartbeat = 0
        self.nr_threads = 2

    def _clear_queue(self, queue_name):
        conn = pika.BlockingConnection(pika.ConnectionParameters(
            host=self._host,
            # port=self._port,
            heartbeat=self._heartbeat,
            blocked_connection_timeout=None,
            virtual_host='/',
            credentials=pika.PlainCredentials(self._usrname, self._usrname)
        ))
        channel = conn.channel()
        channel.queue_delete(queue=queue_name)
        channel.close()
        conn.close()

    def listen(self, reset_pipe=False):
        assert self.nr_threads is not None
        if reset_pipe:
            printgreen(info_prefix(), 'Reset existing pipes.')
            print('task_pipe_name: ', self._task_queue_name)
            print('result_pipe_name: ', self._result_queue_name)
            self._clear_queue(self._task_queue_name)
            self._clear_queue(self._result_queue_name)
        threads = ThreadPool(self.nr_threads)
        threads.map(self._listen_thread, range(self.nr_threads))
    
    def _listen_thread(self, thread_idx):
        conn = pika.BlockingConnection(pika.ConnectionParameters(
            host=self._host,
            # port=self._port,
            heartbeat=self._heartbeat,
            blocked_connection_timeout=None,
            virtual_host='/',
            credentials=pika.PlainCredentials(self._usrname, self._usrname)
        ))
        task_channel = conn.channel()
        task_channel.queue_declare(queue=self._task_queue_name)
        result_channel = conn.channel()
        result_channel.queue_declare(queue=self._result_queue_name)

        def fail(*args, **kwargs):
            print(args)
            print(kwargs)
            raise NotImplementedError

        result_channel.add_on_cancel_callback(fail)
        task_channel.basic_qos(prefetch_count=1)
        task_channel.basic_consume(
            functools.partial(self._solve_task_callback, result_channel=result_channel), 
            queue=self._task_queue_name)
        print(info_prefix(), 'Listening %d' % thread_idx)
        print()
        task_channel.start_consuming()

    def _solve_task_callback(self, cur_channel, frame, properties, body, result_channel):
        data = pickle.loads(body)
        task_key, task = data
        printgreen(info_prefix(), 'receive task: ', task_key)
        try:
            result = self.eval(task)
        except:
            import traceback
            traceback.print_exc()
            time.sleep(5)
            os._exit(1)
            return {'status': 'uncatched error'}
        printgreen(info_prefix(), 'finished evaluate')
        del task
        obj = pickle.dumps((task_key, result))
        result_channel.basic_publish(exchange='', routing_key=self._result_queue_name, body=obj)
        cur_channel.basic_ack(delivery_tag=frame.delivery_tag)

    def run(self, threads, *, reset_pipe=False):
        self.nr_threads = threads
        self.listen(reset_pipe=reset_pipe)

    def eval(self, task):
        res = dict()
        try:
            # test_accuracy = evaluate_path(task)
            test_accuracy = evaluate_operations(task)
            res = {'status': 'success', 'accuracy': test_accuracy}
            return res
        except:
            import traceback
            traceback.print_exc()
            res['status'] = 'failure'
            os._exit(1)
            return res


if __name__ == "__main__":
    gpu_server = GpuServer()
    gpu_server.run(threads=1)
