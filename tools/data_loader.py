import copy
import torch.multiprocessing as mp


class DataLoader(object):
    def __init__(self, dataset,node_rank, world_size):
        self.manager = mp.Manager()
        self.queue = self.manager.Queue()
        self.lock = mp.Lock()

        todo_count = 0

        # 将任务加入任务队列
        for prob_idx, prob in enumerate(dataset):
            # 在分布式环境中为每个节点均匀分配任务，world_size是节点总数，node_rank是当前节点编号，通过 % 把任务分配给对应的节点
            if todo_count % world_size == node_rank:
                self.queue.put((prob_idx, copy.deepcopy(prob)))
            todo_count += 1

        print('Number of TODO Problems: {}'.format(self.queue.qsize()))
    
    def size(self):
        return self.queue.qsize()
    
    def get(self):
        with self.lock:
            if self.queue.qsize() > 0:
                return self.queue.get()
        return None,None
