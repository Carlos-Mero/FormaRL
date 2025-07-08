import time
import torch.multiprocessing as mp


class CompilerChecker(mp.Process):
    def __init__(self, idx,data_loader,verifier,response_dict):
        self.idx = idx
        self.data_loader=data_loader
        self.verifier = verifier
        self.response_dict = response_dict
        self.lock = mp.Lock()
        super().__init__()
    
    def process_print(self, message):
        print('Process ID: {:3d}'.format(self.idx), message)

    def run(self):
        while True:
            # 从data_loader中获取问题数据
            prob_idx, data = self.data_loader.get()
            if prob_idx is None: 
                break
            
            candidate_list,request_id_list = [],[]

            candidate =data['flp']
            candidate_list.append(candidate)

            request_id = self.verifier.submit_request(candidate)
            request_id_list.append(request_id)


            verification_start_wait_time = time.time()
    
            result_list = self.verifier.get_all_request_outputs(request_id_list)


            verification_timecost = time.time() - verification_start_wait_time


            success_count = sum([int(result['pass']) for result in result_list])
            
            self.process_print('Success: {} / {}    Problem ID: {}   Verfication: {:.2f} secs'.format(
                success_count, len(candidate_list), prob_idx,verification_timecost,
            ))


            for result in result_list:
                # 把结果加入共享字典
                with self.lock:
                    self.response_dict[f"{prob_idx}"] = (result if not result['pass'] else '')
            