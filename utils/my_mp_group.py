import torch.multiprocessing as mp
import time
import traceback
from typing import Any, List


def job_wrapper(func, args, data_container):
    """
    子进程执行的包装函数，将结果对象添加到共享列表中。
    """
    try:
        out_value = func(*args)
        # 将结果对象添加到 Manager().list() 中
        data_container.append(out_value)
    except Exception as e:
        print("ERROR\n")
        traceback.print_exc()
        print()
        # 发生错误时，向共享列表中添加 None
        data_container.append(None)


class Group():

    def __init__(self):
        # 使用 Manager() 创建一个可共享的列表
        self.manager = mp.Manager()
        self.jobs = []
        self.return_data = self.manager.list()  # 替换为 Manager().list()
        self.callback = []

    def add_job(self, func, args, callback=None):
        ctx = mp.get_context("spawn")
        # 将共享列表传递给 job_wrapper
        self.jobs.append(ctx.Process(target=job_wrapper, args=(func, args, self.return_data)))
        if callback is not None:
            self.callback.append(callback)

    def run_jobs(self, num_proc) -> List[Any]:
        next_job = 0
        num_jobs_open = 0
        jobs_finished = 0
        jobs_open = set()

        while jobs_finished != len(self.jobs):

            # 检查已完成的子进程
            jobs_closed = []
            for job_index in jobs_open:
                if not self.jobs[job_index].is_alive():
                    self.jobs[job_index].join()
                    self.jobs[job_index].terminate()
                    num_jobs_open -= 1
                    jobs_finished += 1
                    jobs_closed.append(job_index)

            for job_index in jobs_closed:
                jobs_open.remove(job_index)

            # 启动新子进程
            while num_jobs_open < num_proc and next_job != len(self.jobs):
                self.jobs[next_job].start()
                jobs_open.add(next_job)
                next_job += 1
                num_jobs_open += 1

            # 等待所有任务完成
            time.sleep(0.1)

        # 转换共享列表为普通列表并返回
        return list(self.return_data)