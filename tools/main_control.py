import pika
import sys
import time
import os
from queue import Queue
import threading
import pickle
import functools
import termcolor
import numpy as np
import datetime
from test_supernet import get_valid_choices
print=functools.partial(print,flush=True)
tostring=lambda *args:' '.join(map(str,args))
printred=lambda *args,**kwargs:termcolor.cprint(tostring(*args),color='red',flush=True,**kwargs)
printgreen=lambda *args,**kwargs:termcolor.cprint(tostring(*args),color='green',flush=True,**kwargs)

def info_prefix():
    return '[{} info]'.format(datetime.datetime(1,1,1).now())

choice = lambda x: x[np.random.randint(len(x))] if isinstance(x, tuple) else choice(tuple(x))
from lib.models.ea_path_finder import EAPathFinder


class config:
    population_num = 50
    select_num = 10
    mutation_num = 20
    m_prob = 0.1
    crossover_num = 20
    max_epochs = 20
    exp_name = 'exp_ea_search'


class MainControl:

    def __init__(self):
        self._task_queue_name = 'task'
        self._result_queue_name = 'result'
        self._host = 'localhost'
        self._usrname = 'test'
        self._heartbeat = 0
        self._buffer_queue = Queue()
        self._buffer = {}
        self._task_idx = 0
        self._thread = None

        # For Evolution Search
        self.path_finder = EAPathFinder()
        self.memory = []
        self.candidates = []
        self.vis_dict = {}
        self.keep_top_k = {config.select_num: [], config.population_num: []}
        self.epoch = 0
        self.checkpoint_name = os.path.join(config.exp_name, 'checkpoint.pkl')
        self.log_file = os.path.join(config.exp_name, 'log.txt')

    def connect(self, reset_pipe=False):
        conn = pika.BlockingConnection(pika.ConnectionParameters(
            host=self._host,
            heartbeat=self._heartbeat,
            blocked_connection_timeout=None,
            virtual_host='/',
            credentials=pika.PlainCredentials(self._usrname, self._usrname)
        ))
        self.conn = conn
        self._task_channel = conn.channel()
        if reset_pipe:
            self._task_channel.queue_delete(self._task_queue_name)
        self._task_channel.queue_declare(self._task_queue_name)

        def start_consuming():
            conn = pika.BlockingConnection(pika.ConnectionParameters(
                host=self._host,
                heartbeat=self._heartbeat,
                blocked_connection_timeout=None,
                virtual_host='/',
                credentials=pika.PlainCredentials(self._usrname, self._usrname)
            ))
            result_channel = conn.channel()
            if reset_pipe:
                result_channel.queue_delete(self._result_queue_name)
            result_channel.queue_declare(self._result_queue_name)
            result_channel.basic_consume(self._fetch_result_callback, queue=self._result_queue_name)
            result_channel.start_consuming()

        if self._thread is not None:
            self._thread = None
        thread = threading.Thread(target=start_consuming)
        thread.start()
        self._thread = thread

    def _fetch_result_callback(self, channel, frame, properties, body):
        data = pickle.loads(body)
        print('Receive result: ', data)
        self._buffer_queue.put(data)
        channel.basic_ack(delivery_tag=frame.delivery_tag)

    def send_task(self, task):
        task_key = str(task.tolist()) if not isinstance(task, str) else task
        body = pickle.dumps((task_key, task))
        while True:
            try:
                self._task_channel.basic_publish(exchange='', routing_key=self._task_queue_name, body=body)
                break
            except:
                import traceback
                traceback.print_exc()
                time.sleep(10)
                printred(' Send failed, reconnecting>>>')
                self.connect()
        return task_key

    def get_result(self, task_key, *, timeout=1000):
        task_key = task_key if isinstance(task_key, str) else str(task_key.tolist())
        if task_key in self._buffer:
            task_key = task_key if isinstance(task_key, str) else str(task_key.tolist()) 
            result = self._buffer[task_key]
            return result
        while True:
            try:
                cur_key, cur_result = self._buffer_queue.get(timeout=timeout)
            except Exception as e:
                print(e)
                continue
            self._buffer[cur_key] = cur_result
            if cur_key == task_key:
                return cur_result

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_name):
            return False
        with open(self.checkpoint_name, 'rb') as fin:
            info = pickle.load(fin)
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']
        self._buffer = info['buffer']
        print('==> Load checkpoint from ', self.checkpoint_name)
        return True
    
    def save_checkpoint(self):
        if not os.path.exists(config.exp_name):
            os.makedirs(config.exp_name)
        info = {
            'memory': self.memory,
            'candidates': self.candidates,
            'vis_dict': self.vis_dict,
            'keep_top_k': self.keep_top_k,
            'epoch': self.epoch,
            'buffer': self._buffer
        }
        with open(self.checkpoint_name, 'wb') as fout:
            pickle.dump(info, fout, pickle.HIGHEST_PROTOCOL)
        print('==> Save checkpoint to ', self.checkpoint_name)

    def legal(self, cand, is_encoded_type=False):
        if is_encoded_type:
            assert isinstance(cand, list)
            if cand.count(2) != 2:
                return False
            if cand.count(1) != 3:
                return False
            check_cand = self.decode_path(cand)
        else:
            check_cand = cand
        assert isinstance(check_cand, str)
        if check_cand not in self.vis_dict:
            self.vis_dict[check_cand] = {}
        info = self.vis_dict[check_cand]
        if 'visited' in info:
            return False
        return True

    def random_can(self, num):
        print('==> Random Select ....')
        candidates = []
        while len(candidates) < num:
            path_str = str(self.path_finder.choice_random_path().tolist())
            self.vis_dict[path_str] = {}
            if not self.legal(path_str):
                continue
            candidates.append(path_str)
            print('=> Random %d / %d' % (len(candidates), num))
        return candidates

    def sync_candidates(self):
        sync_num = 0
        while True:
            ok = True
            for cand in self.candidates:
                info = self.vis_dict[cand]
                if 'accuracy' in info:
                    continue
                ok = False
                if 'task_key' not in info:  # not send this task
                    info['task_key'] = self.send_task(np.array(eval(cand)))  # Send task
            self.save_checkpoint()
            for cand in self.candidates:
                info = self.vis_dict[cand]
                if 'accuracy' in info:
                    continue
                key = info.pop('task_key')
                try:
                    res = self.get_result(key, timeout=1800)
                    info['accuracy'] = res['accuracy']
                    info['error'] = 1 - info['accuracy']
                    self.save_checkpoint()
                    printgreen(info_prefix(), 'Sync num: %d' % sync_num)
                    sync_num += 1
                except:
                    import traceback
                    traceback.print_exc()
                    time.sleep(1)
            time.sleep(5)
            if ok:
                break

    def encoder_path(self, path):
        path = np.array(eval(path))
        cand = []
        for i in range(1, len(path)):
            cur_node = path[i]
            pre_node = path[i - 1]
            gap = cur_node - pre_node
            if (gap == np.array([0, 0])).all():
                cand.append(0)
            elif (gap == np.array([1, 0])).all():
                cand.append(1)
            elif (gap == np.array([1, 1])).all():
                cand.append(2)
            else:
                raise NotImplementedError
        return cand

    def decode_path(self, e_path):
        cand = [np.array([0, 0])]
        for p in e_path:
            if p == 0:
                cand.append(cand[-1] + np.array([0, 0]))
            elif p == 1:
                cand.append(cand[-1] + np.array([1, 0]))
            elif p == 2:
                cand.append(cand[-1] + np.array([1, 1]))
            else:
                raise NotImplementedError
        res = np.stack(cand)
        res = str(res.tolist())
        return res

    def update_top_k(self, candidates, *, k, key, reverse=False):
        assert k in self.keep_top_k, 'k=%s, keep_topk_keys=%s' % (str(k), str(self.keep_top_k.keys()))
        print('==> Select ....')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k, 'k=%s, keep_topk_keys=%s' % (str(k), str(self.keep_top_k.keys()))
        print('==> Mutation ....')
        max_iters = 10 * mutation_num
        res = []
        while len(res) < mutation_num and max_iters > 0:
            cand = choice(self.keep_top_k[k])
            e_cand = self.encoder_path(cand)
            for i in range(len(e_cand)):
                if np.random.random_sample() < m_prob and e_cand[i] != 0:
                    choice_i = np.random.randint(len(e_cand))
                    e_cand[choice_i], e_cand[i] = e_cand[i], e_cand[choice_i]
            cand = self.decode_path(e_cand)
            if not self.legal(cand):
                continue
            res.append(cand)
            print('Mutation %d/%d, path: %s' % (len(res), mutation_num, cand))
            max_iters -= 1
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k, 'k=%s, keep_topk_keys=%s' % (str(k), str(self.keep_top_k.keys()))
        print('==> CrossOver ....')
        res = []
        max_iter = 20 * crossover_num
        while len(res) < crossover_num and max_iter > 0:
            cand1 = choice(self.keep_top_k[k])
            e_cand1 = self.encoder_path(cand1)
            cand2 = choice(self.keep_top_k[k])
            e_cand2 = self.encoder_path(cand2)
            cross_cand = [choice([i, j]) for i, j in zip(e_cand1, e_cand2)]
            if not self.legal(cross_cand, is_encoded_type=True):
                continue
            cand = self.decode_path(cross_cand)
            res.append(cand)
            print('Crossover %d/%d, path: %s' % (len(res), crossover_num, cand))
            max_iter -= 1
        return res

    def train(self):
        if not os.path.exists(config.exp_name):
            os.makedirs(config.exp_name)
        fout_log = open(self.log_file, 'w')

        print('population_num=%d, select_num=%d, mutation_num=%d '
              'crossover_num=%d, random_num=%d, max_epochs=%d' % (
                  config.population_num, config.select_num, config.mutation_num, 
                  config.crossover_num, config.population_num - config.mutation_num - config.crossover_num,
                  config.max_epochs
              ))
        if not self.load_checkpoint():
            self.candidates = self.random_can(config.population_num)
            self.save_checkpoint()

        while self.epoch < config.max_epochs:
            printgreen(info_prefix(), 'Epoch: ', self.epoch)
            self.sync_candidates()
            printgreen(info_prefix(), 'Synth finished')
            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)
                self.vis_dict[cand]['visited'] = True

            printgreen(info_prefix(), 'Update top %d' % config.select_num)
            self.update_top_k(self.candidates, k=config.select_num, key=lambda x: self.vis_dict[x]['error'])
            printgreen(info_prefix(), 'Update top %d' % config.population_num)
            self.update_top_k(self.candidates, k=config.population_num, key=lambda x: self.vis_dict[x]['error'])
            self.epoch += 1

            # Print top1 result every epoch
            top1_key = self.keep_top_k[config.population_num][0]
            log_ = info_prefix() + ' Epoch %d, No1. result %.5f, path %s\n' % (self.epoch, self.vis_dict[top1_key]['accuracy'], top1_key)
            printred(log_)
            fout_log.write(log_)
            fout_log.flush()

            mutation = self.get_mutation(config.select_num, config.mutation_num, config.m_prob)
            crossover = self.get_crossover(config.select_num, config.crossover_num)
            rand = self.random_can(config.population_num - len(mutation) - len(crossover))
            self.candidates = mutation + crossover + rand
        printgreen(info_prefix(), 'Finished evolution training!')
        fout_log.close()


def random_search(random_num=500):
    control = MainControl()
    control.connect(reset_pipe=True)
    valid_op_choices = get_valid_choices()
    accuracy_dict = {}
    while len(accuracy_dict) < random_num:
        op_choices = [np.random.randint(x) for x in valid_op_choices]
        op_choices_key = str(op_choices)
        if op_choices_key in accuracy_dict:
            continue
        task_key = control.send_task(np.array(op_choices))
        accuracy_dict[task_key] = {}
        print('Send task %d/%d: %s' % (len(accuracy_dict), random_num, task_key))
    
    result_txt = 'find_ops_coco.txt'
    max_accuracy = 0

    with open(result_txt, 'w') as fout:
        for k, v in accuracy_dict.items():
            printred('Waitting key: ', k)
            task_result = control.get_result(k)
            accuracy_dict[k] = task_result
            test_accuracy, flops, params = task_result['accuracy']
            
            if test_accuracy > max_accuracy:
                max_accuracy = test_accuracy
                task_result['status'] = '*'
            else:
                task_result['status'] = ''
            printgreen(k, ', acc: %.4f, max %.4f' % (test_accuracy, max_accuracy))
            fout.write('#acc: %s  #path: %s\n' % (task_result, k))
            fout.flush()


def evolution_search():
    control = MainControl()
    control.connect(reset_pipe=True)
    control.train()


if __name__ == "__main__":
    random_search(1000)
    # evolution_search()
    # control = MainControl()
    # control.candidates = [str(control.path_finder.choice_random_path().tolist()) for i in range(50)]
    # control.keep_top_k[50] = control.candidates
    # # control.get_mutation(50, 10, 0.1)
    # control.get_crossover(50, 10)
