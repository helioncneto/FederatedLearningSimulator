import copy
import pickle
import random
import numpy as np
from typing import Tuple, List

__all__ = ['selection_ig', 'update_participants_score', 'calc_ig', 'load_from_file']


def selection_ig(selected_participants_num: int, ep_greedy: float, not_selected_participants: List[int],
                 participants_score: dict, participants_count: dict = {}) -> tuple:
    selection_helper = copy.deepcopy(participants_score)
    selected_participants = []
    for _ in range(selected_participants_num):
        p = random.random()
        if p < ep_greedy:
            print("Random selection")
            if len(not_selected_participants) != 0:
                selected = np.random.choice(not_selected_participants)
                not_selected_participants.remove(selected)
            else:
                selected = np.random.choice(list(selection_helper.keys()))
                if len(participants_count.keys()) != 0:
                    while participants_count[selected] >= 3:
                        print(f'Participant {selected} is blocked')
                        selected = np.random.choice(list(selection_helper.keys()))

            selection_helper.pop(selected)
            selected_participants.append(selected)
        else:
            # Select the best participant
            print("Greedy selection")
            sel = 0
            selected = sorted(selection_helper, key=selection_helper.get, reverse=True)[sel]
            while participants_count[selected] >= 3:
                print(f'Participant {selected} is blocked')
                sel += 1
                selected = sorted(selection_helper, key=selection_helper.get, reverse=True)[sel]
            if selected in not_selected_participants:
                not_selected_participants.remove(selected)
            selection_helper.pop(selected)
            selected_participants.append(selected)
    return selected_participants, not_selected_participants


def update_participants_score(participants_score: dict, cur_ig: dict, ig: dict,
                              eg_momentum: float = 0.9) -> Tuple[dict, dict]:
    for client_id, client_ig in cur_ig.items():
        if client_id not in ig.keys():
            ig[client_id] = []
            ig[client_id].append(client_ig)
        else:
            ig[client_id].append(client_ig)
        if len(ig[client_id]) <= 1:
            participants_score[client_id] = ig[client_id][0]
        else:
            delta_term = sum(ig[client_id][:-1]) / len(ig[client_id][:-1])
            participants_score[client_id] = ((1 - eg_momentum) * delta_term) + (eg_momentum * ig[client_id][-1])
    return participants_score, ig


def calc_ig(parent_entropy: float, child_entropy: dict, w: dict) -> dict:
    ig = {}
    for idx, (client_id, child) in enumerate(child_entropy.items()):
        #w = child_size[idx]/parent_size
        #curr_ig = -np.log(parent_entropy) - np.log(w * child)
        #curr_ig = -np.log(parent_entropy) - w * (-np.log(child))
        curr_ig = -np.log(parent_entropy) - w[client_id] * np.log(child)
        ig[client_id] = curr_ig
    return ig


def load_from_file(path: str) -> dict:
    f = open(path, "rb")
    output = pickle.load(f)
    f.close()
    return output


def save_dict_file(path: str, dictionary_data: dict) -> None:
    f = open(path, "wb")
    pickle.dump(dictionary_data, f)
    f.close()