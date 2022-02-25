from __future__ import annotations

import csv
import json
from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Generator, Set, Union

import numpy as np
from faker import Faker

ACTIONS = ['like', 'dislike', 'repost', 'comment', 'post', ]
ACTIONS_WEIGHTS = np.array([10, 5, 5, 2, 1])
ACTION_PROBABILITIES = ACTIONS_WEIGHTS / np.sum(ACTIONS_WEIGHTS)
DEVICE_TYPE = ['Android', 'IPhone', 'IPad', 'Windows', 'Linux', 'Macintosh']

HEADER = ['Timestamp', 'IP', 'Device', 'Action', 'Description']

fake = Faker()


class User:
    ip_addr: str
    device: str

    def __init__(self) -> None:
        self.ip_addr = fake.unique.ipv4()
        self.device = fake.random_element(DEVICE_TYPE)


class DataGenerator:
    def __init__(
        self,
        user_num: int,
        samples: int,
        start_date: datetime,
        offset: float
    ) -> None:
        self.users = [User() for _ in range(user_num)]

        user_weights = np.abs(
            np.random.normal(loc=100, size=user_num))
        self.num_of_anomalyous_users = np.random.randint(
            int(user_num * 0.01),
            int(user_num * 0.05) + 1
        )

        self.anomalous_users_idxs = np.random.choice(
            user_num,
            self.num_of_anomalyous_users,
            replace=False
        )

        user_weights[self.anomalous_users_idxs] *= 100

        self.user_probabilities = user_weights / np.sum(user_weights)

        self.num_of_samples = samples
        self.date = start_date
        self.offset = offset

        self.analysis_results: Dict[str, Any] = {
            'action_count': Counter(),
            'action_device': defaultdict(Counter),
            'user_action_count': defaultdict(Counter),
        }

    @cached_property
    def anomalous_users(self) -> Set[str]:
        return set(map(
            lambda x: x.ip_addr,
            np.array(self.users)[self.anomalous_users_idxs]
        ))

    def gen_timestamp(self) -> str:
        offset = timedelta(
            milliseconds=abs(np.random.normal(self.offset, 2.5))
        )
        self.date += offset
        return self.date.isoformat()

    def generate(self) -> Generator[Dict[str, Union[str, float]], None, None]:
        for _ in range(self.num_of_samples):
            user: User = np.random.choice(
                self.users,  # type: ignore
                1,
                p=self.user_probabilities
            )[0]
            timestamp = self.gen_timestamp()
            action: str = np.random.choice(
                ACTIONS, 1, p=ACTION_PROBABILITIES
            )[0]

            yield {
                'Timestamp': timestamp,
                'IP': user.ip_addr,
                'Device': user.device,
                'Action': action,
                'Description': fake.sentence(10),
            }

    def analyse(self, data: Dict[str, Union[str, float]]) -> None:
        if data['IP'] in self.anomalous_users:
            return

        self.analysis_results['action_count'] += Counter({data['Action']: 1})
        self.analysis_results['action_device'][data['Action']] += Counter(
            {data['Device']: 1})
        self.analysis_results['user_action_count'][data['Action']] += Counter(
            {data['IP']: 1}
        )

    def gen_report(self) -> Dict:
        out: dict[str, Any] = {}

        action_count = dict(
            self.analysis_results['action_count'].most_common())
        action_device = {
            k: dict(
                v.most_common()) for k,
            v in self.analysis_results['action_device'].items()}
        user_action_count = {
            k: dict(
                v.most_common()) for k,
            v in self.analysis_results['user_action_count'].items()}

        out['anomalous_users_count'] = self.num_of_anomalyous_users
        out['anomalous_users'] = list(self.anomalous_users)
        out['action_count'] = action_count
        out['action_device'] = action_device
        out['user_action_count'] = user_action_count

        return out


def datetime_from_iso(date: str) -> datetime:
    return datetime.fromisoformat(date)


parser = ArgumentParser()
parser.add_argument(
    '-u',
    '--users',
    nargs='?',
    type=int,
    default=200,
    help='number of users, which will be "using" the social media'
)
parser.add_argument(
    '-s',
    '--samples',
    nargs='?',
    type=int,
    default=5000,
    help='number of samples to generate'
)
parser.add_argument(
    '-o',
    '--offset',
    nargs='?',
    type=int,
    default=100,
    help='max time between each sample'
)
parser.add_argument('output', nargs='?', type=Path, help='file path, where results will be stored')


if __name__ == '__main__':
    options = parser.parse_args()
    gen = DataGenerator(
        options.users,
        options.samples,
        datetime.now(),
        options.offset
    )
    with open(options.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, HEADER)
        writer.writeheader()

        for i in gen.generate():
            gen.analyse(i)
            writer.writerow(i)

    Path('analysis.json').write_text(
        json.dumps(gen.gen_report(), indent=2),
        'utf-8'
    )
