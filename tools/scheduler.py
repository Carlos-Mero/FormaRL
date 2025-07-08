import time
import ctypes
import threading #多线程模块
import multiprocessing as mp #支持多进程编程。允许创建并行执行的进程
import numpy as np


# 管理任务队列的多线程安全类：批量获取任务、监控任务状态、任务完成后关闭队列
class TaskQueue(object):
    def __init__(self, batch_size=512, name='test'):
        self.name = name
        self.batch_size = batch_size
        self.manager = mp.Manager() # 管理多进程共享的数据结构，如list dict
        self.waiting_list = self.manager.list() # 创建多进程安全列表，存储待处理的任务
        self.all_tasks_done = mp.Event() #创建多进程安全对象，标记所有任务是否已完成
        self.lock = mp.Lock() # 创建多进程安全的锁，保护对共享资源的访问

        self._monitor_log = self.manager.list() #创建多进程安全列表，存储监控日志
        self._monitor_thread = threading.Thread(target=self._monitor) #创建一个线程，监控任务队列的状态
        self._monitor_thread.start() # 启动该线程
    
    # 监控方法，定期打印任务队列的状态
    def _monitor(self):
        last_log_time = time.time() # 记录上一次打印日志的时间

        # 循环检查all_tasks_done是否被设置，若是未被设置（未完成），继续监控
        while not self.all_tasks_done.is_set():
            # 检查距离上一次打印日志是否已经超过1min
            if time.time() - last_log_time >= 60.0:
                # 超过，获取锁，确保对共享资源的访问是线程安全的
                with self.lock:
                    # 检查监控日志列表是否非空 
                    if len(self._monitor_log) > 0:
                        #非空，打印任务队列状态：队列名称 过去一段时间从队列中获取的任务数量 平均贝茨获取的任务数量 当前队列中等待的任务数量
                        print('TaskQueue-{}:  {} requests popped with avg batch_size {:.1f} in last period  {} waiting in queue'.format(
                            self.name, np.sum(self._monitor_log), np.mean(self._monitor_log), len(self.waiting_list),
                        ))
                        self._monitor_log[:] = [] # 清空监控日志列表
                last_log_time = time.time() # 更新为当前时间
            time.sleep(1.0) # 线程休眠1s，避免占用过多CPU资源
            #print("Sleeping for 1 sec......")
    
    # 获取队列长度
    def __len__(self):
        return len(self.waiting_list)
    
    # 添加任务
    def put(self, item):
        # 确保对待处理的任务列表的访问是线程安全的
        with self.lock:
            self.waiting_list.append(item)
    
    # 获取任务
    def get(self, no_wait=False):
        # all_tasks_done未被设置，继续获取任务
        while not self.all_tasks_done.is_set():
            # 获取锁。确保对待处理的任务列表的访问是线程安全的
            with self.lock:
                # 检查待处理的任务列表是否为空
                if len(self.waiting_list) > 0:
                    tasks = self.waiting_list[:self.batch_size] # 从waiting_list中获取batch_size个任务
                    self.waiting_list[:self.batch_size] = [] # 从waiting_list中移除这些任务
                    self._monitor_log.append(len(tasks)) # 将本次获取的任务数量记录到监控日志列表中
                    return tasks # 返回获取的任务
            # 没有任务时立即退出循环
            if no_wait:
                break

            time.sleep(0.1) # 休眠
        return None
    
    # 关闭队列
    def close(self):
        self.all_tasks_done.set() # 设置all_tasks_done
        self._monitor_thread.join() # 等待监控线程结束

# 管理任务调度和结果收集
class ProcessScheduler(object):
    def __init__(self, batch_size=512, name='test'):
        self.name = name
        self.manager = mp.Manager()
        self.batch_size = batch_size
        self.task_queue = TaskQueue(batch_size=batch_size, name=name) # 创建管理任务队列的对象
        self.request_statuses = self.manager.dict() # 创建多进程安全的字典，存储每个请求的状态
        self.request_counter = mp.Value(ctypes.c_int32, 0) # 创建多进程安全的整数值，用于生成唯一的请求ID
        self.lock = mp.Lock() # 创建多进程安全的锁

    # 提交单个请求的方法
    # params: 要提交的任务数据data
    def submit_request(self, data):
        with self.lock:
            # 生成唯一的请求ID
            self.request_counter.value += 1 
            request_id = self.request_counter.value

            self.request_statuses[request_id] = None # 初始化当前请求状态为None
            self.task_queue.put((time.time(), request_id, data)) # 添加任务 (当前时间戳，请求ID，任务数据)
        return request_id # 返回生成的请求ID
    
    #提交多个请求，调用提交单个请求方法实现
    def submit_all_request(self, data_list):
        request_id_list = [self.submit_request(data) for data in data_list]
        return request_id_list

    # 获取请求状态
    # params: 要查询的请求ID request_id
    def get_request_status(self, request_id):
        with self.lock:
            response = self.request_statuses.get(request_id, None)
            if response is not None:
                self.request_statuses.pop(request_id) # 移除请求ID
            return response
    
    # 获取请求输出
    # params: 要输出的请求ID request_id
    def get_request_outputs(self, request_id):
        start_time = time.time()
        timeout = 300
        # 进入循环，等待请求完成
        while True:
            outputs = self.get_request_status(request_id)
            if outputs is not None:
                #print("Successfully get response!!")
                return outputs
            if time.time() - start_time > timeout:
                #print("--Time Limit Exceed!!")
                return {
                "pass": False,
                "complete": False,
                "system_errors": "Request timed out after 300 seconds",
                "system_messages": "Timeout"
            }                

            time.sleep(1.0)
    
    # 获取所有请求输出
    def get_all_request_outputs(self, request_id_list):
        outputs_list = []
        for request_id in request_id_list:
            outputs_list.append(self.get_request_outputs(request_id))
        return outputs_list
    
    # 关闭调度器
    def close(self):
        self.task_queue.close()
