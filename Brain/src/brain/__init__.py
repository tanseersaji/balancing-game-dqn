from collections import deque
import torch
import numpy as np
import socket
import random
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from datetime import datetime

torch.set_default_device("cpu")

HOST = "127.0.0.1"
PORT = 5555
DEAD_THRESHOLD = 0.01
CONTROLS = [0,1,2]

EPSILON_DECAY = 0.995
EPSILON_MIN = 0.05
BATCH_SIZE = 128
GAMMA = 0.99
TARGET_UPDATE_FREQUENCY = 5

writer = SummaryWriter('runs/balance_experiment_2')

class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = torch.nn.Linear(6, 128)
        self.l2 = torch.nn.Linear(128, 128)
        self.l3 = torch.nn.Linear(128, 128)
        self.l4 = torch.nn.Linear(128, 3)
        self.activation = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l1 = self.l1(x)
        l1 = self.activation(l1)
        l2 = self.l2(l1)
        l2 = self.activation(l2)
        l3 = self.l3(l2)
        l3 = self.activation(l3)
        return self.l4(l3)

def optimize_model(model:Model, target_model:Model, optimiser:torch.optim.Adam, memory, eval_mode:bool=False):
    if len(memory) < BATCH_SIZE:
        return -1000.0

    transactions = random.sample(memory, BATCH_SIZE)

    state_batch = torch.stack([t[0] for t in transactions])
    action_batch = torch.tensor([t[1] for t in transactions],dtype=torch.int64).unsqueeze(1)
    reward_batch = torch.tensor([t[2] for t in transactions], dtype=torch.float32)
    next_state_batch = torch.stack([t[3] for t in transactions])
    done_batch = torch.tensor([t[4] for t in transactions], dtype=torch.float32)

    state_action_values = model(state_batch).gather(1, action_batch).squeeze()

    with torch.no_grad():
        next_state_values = target_model(next_state_batch).max(1)[0]
        expected_state_action_values = reward_batch + (GAMMA * next_state_values * (1 - done_batch))

    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    optimiser.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
    optimiser.step()

    return loss.item()


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--new", action="store_true", help="Start new training", default=False)
    parser.add_argument("--old", action="store_true", help="Continue training", default=False)
    args = parser.parse_args()

    if args.old:
        args.new = False

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)

    epsilon = 1.0

    last_state = None
    last_action = None
    best_session_reward = 52500.0
    session_reward = 0.0

    model = Model()

    if os.path.exists("best_model.pth"):

        if args.new:
            os.rename("best_model.pth", f"best_model_old_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
            print("Starting new training")
        else:
            print("Continuing training")
            model.load_state_dict(torch.load("best_model.pth"))
            epsilon = EPSILON_MIN
            print("Loaded best model")

    target_model = Model()
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    optimiser = torch.optim.Adam(model.parameters())
    memory = deque(maxlen=500000)
    generation_count = 0

    while True:
        print("Waiting for Unity Agent")

        conn, _ = server.accept()

        print("Connected...")

        while True:
            data = conn.recv(1024).decode("utf-8").strip()
            if not data:
                print("Connection Dropped")
                break

            raw_state = [float(x) for x in data.split(",")]

            normalised_state = [
                raw_state[0] / 10.0,
                raw_state[1] / 10.0,
                raw_state[2] / 10.0,
                raw_state[3] / 10.0,
                raw_state[4] / 180.0,
                raw_state[5] / 100.0
            ]

            state = torch.tensor(normalised_state, dtype=torch.float32)
            death_flag = int(raw_state[-1])

            reward = -100.0 if death_flag == 1 else 1.0
            session_reward += reward

            if last_state is not None:
                memory.append((last_state, last_action, reward, state, death_flag))

            loss = optimize_model(model, target_model, optimiser, memory)

            if session_reward > best_session_reward and session_reward % 500 == 0:
                best_session_reward = session_reward
                print(f"New best session reward: {best_session_reward}")
                torch.save(model.state_dict(), "best_model.pth")

            if death_flag == 1:
                last_state = None
                last_action = 0
                generation_count += 1
                
                if generation_count % TARGET_UPDATE_FREQUENCY == 0:
                    print(f"Updating target model -- Generation: {generation_count} -- Loss: {loss} -- Epsilon: {epsilon} -- Session Reward: {session_reward}")
                    target_model.load_state_dict(model.state_dict())

                writer.add_scalar('Training/Loss', loss, generation_count)
                writer.add_scalar('Training/SessionReward', session_reward, generation_count)

                if best_session_reward < session_reward:
                    best_session_reward = session_reward
                    print(f"New best session reward: {session_reward}")
                    torch.save(model.state_dict(), "best_model.pth")

                epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)
                session_reward = 0.0

                conn.sendall("0\n".encode("utf-8"))
                continue

            if np.random.random() < epsilon:
                action = np.random.choice(CONTROLS)
            else:
                with torch.no_grad():
                    action = int(model(state).argmax().item())

            last_state = state
            last_action = action


            conn.sendall(f"{action}\n".encode("utf-8"))