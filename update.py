# Copyright 2025 Jason Deng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import pandas as pd
import github
import os
import time
import schedule
from datetime import datetime


class GitHubUpdater:
    def __init__(self, token: str, repo_name: str, file_path: str = "signal_df.csv"):
        """
        初始化GitHub更新器
        
        :param token: GitHub访问令牌
        :param repo_name: 仓库名称，格式为 "用户名/仓库名"
        :param file_path: 要更新的文件路径
        """
        self.token = token
        self.repo_name = repo_name
        self.file_path = file_path
        self.github_client = github.Github(token)
        self.repo = self.github_client.get_repo(repo_name)
    
    def update_signal_file(self, local_file_path: str = "signal_df.csv"):
        """
        更新signal_df.csv文件到GitHub仓库
        
        :param local_file_path: 本地文件路径
        """
        try:
            # Read local file content
            with open(local_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            file_name = local_file_path.split('\\')[-1]
            self.file_path = f'{file_name}'
            # Try to get existing file
            try:
                existing_file = self.repo.get_contents(self.file_path)
                # Update existing file
                self.repo.update_file(
                    path=self.file_path,
                    message=f"Auto update signal data - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    content=content,
                    sha=existing_file.sha
                )
                print(f"Successfully updated file {self.file_path} to repository {self.repo_name}")
            except github.UnknownObjectException:
                # File does not exist, create new file
                self.repo.create_file(
                    path=self.file_path,
                    message=f"Create signal data file - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    content=content
                )
                print(f"Successfully created file {self.file_path} to repository {self.repo_name}")
                
        except Exception as e:
            print(f"Error occurred while updating file: {e}")
    
    def schedule_update(self, interval_minutes: int = 30):
        """
        设置定时更新任务
        
        :param interval_minutes: 更新间隔（分钟）
        """
        schedule.every(interval_minutes).minutes.do(self.update_signal_file)
        
        print(f"已设置定时任务，每 {interval_minutes} 分钟更新一次")
        
        while True:
            schedule.run_pending()
            time.sleep(1)


def main():
    # 配置参数
    GITHUB_TOKEN = os.getenv("TOKEN", "")
    REPO_NAME = 'Steph-hhhhh/factor_dashboard'#"Deng-Alpha/factor_csv"  # 替换为实际的仓库名
    
    # 创建更新器实例
    updater = GitHubUpdater(GITHUB_TOKEN, REPO_NAME)
    data_path = 'C:\\Users\\boyu.deng\\Desktop\\d1\\signal_data'
    # 读取指定路径中的所有.csv文件地址
    csv_files = []
    if os.path.exists(data_path):
        for file_name in os.listdir(data_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(data_path, file_name)
                csv_files.append(file_path)
        print(f"找到 {len(csv_files)} 个CSV文件:")
        for csv_file in csv_files:
            print(f"  - {csv_file}")
    else:
        print(f"路径不存在: {data_path}")
    # 立即执行一次更新
    for csv_file in csv_files:
        start_time = time.time()
        updater.update_signal_file(csv_file)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"处理文件 {os.path.basename(csv_file)} 耗时: {elapsed_time:.2f} 秒")
    
    updater.schedule_update(interval_minutes=1)


def update_once():
    """
    执行一次更新操作的函数
    每次调用都会读取指定路径中的所有CSV文件并上传到GitHub
    """
    # 配置参数
    GITHUB_TOKEN = os.getenv("TOKEN", "")
    REPO_NAME = 'Steph-hhhhh/factor_dashboard'
    
    # 创建更新器实例
    updater = GitHubUpdater(GITHUB_TOKEN, REPO_NAME)
    data_path = 'C:\\Users\\boyu.deng\\Desktop\\d1\\signal_data'
    
    # 读取指定路径中的所有.csv文件地址
    csv_files = []
    if os.path.exists(data_path):
        for file_name in os.listdir(data_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(data_path, file_name)
                csv_files.append(file_path)
        print(f"找到 {len(csv_files)} 个CSV文件:")
        for csv_file in csv_files:
            print(f"  - {csv_file}")
    else:
        print(f"路径不存在: {data_path}")
        return
    
    # 执行一次更新
    for csv_file in csv_files:
        start_time = time.time()
        updater.update_signal_file(csv_file)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Processing file {os.path.basename(csv_file)} took: {elapsed_time:.2f} seconds")
    
    print("Single update operation completed")



if __name__ == "__main__":
    main()
