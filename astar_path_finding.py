import heapq
import time
import numpy as np
from collections import deque
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic: estimated cost from current node to goal
        self.parent = None  # Parent node

    def __lt__(self, other):
        return (self.g + self.h) < (other.g + other.h)

def get_neighbors(map, node):
    neighbors = []
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    for dx, dy in directions:
        nx, ny = node.x + dx, node.y + dy
        if 0 <= nx < len(map) and 0 <= ny < len(map[0]) and map[nx][ny] == 0:
            neighbors.append(Node(nx, ny))
    return neighbors

def heuristic(node, goal):
    # Using Manhattan distance
    return abs(node.x - goal.x) + abs(node.y - goal.y)

def disstance(start, goal):
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

def distance(pos1:int, pos2:int, width:int)->int:
    return abs(pos1 // width - pos2 // width) + abs(pos1 % width - pos2 % width)

def next_pos(pos_x:int, pos_y:int, dir:int, width:int):
    if dir == 0:
        next_pos_x = pos_x
        next_pos_y = pos_y - 1
        if next_pos_y < 0:
            return None
        else:
            return next_pos_x, next_pos_y
    elif dir == 1:
        next_pos_x = pos_x + 1
        next_pos_y = pos_y
        if next_pos_x >= width:
            return None
        else:
            return next_pos_x, next_pos_y
    elif dir == 2:
        next_pos_x = pos_x
        next_pos_y = pos_y + 1
        if next_pos_y >= width:
            return None
        else:
            return next_pos_x, next_pos_y
    elif dir == 3:
        next_pos_x = pos_x - 1
        next_pos_y = pos_y
        if next_pos_x < 0:
            return None
        else:
            return next_pos_x, next_pos_y
    else:
        raise ValueError(f"Invalid direction {dir}")

def reconstruct_path(node):
    path = []
    while node is not None:
        path.append(node)
        node = node.parent
    return path[::-1]

def astar(map, start, goal, max_iter=100):
    start_node = Node(*start)
    goal_node = Node(*goal)

    open_list = []
    heapq.heappush(open_list, start_node)
    closed_list = []

    iteration = 0
    while len(open_list) > 0 and iteration < max_iter:
        current_node = heapq.heappop(open_list)
        closed_list.append(current_node)

        if current_node.x == goal_node.x and current_node.y == goal_node.y:
            return reconstruct_path(current_node)

        neighbors = get_neighbors(map, current_node)
        for neighbor in neighbors:
            if neighbor in closed_list:
                continue

            neighbor.g = current_node.g + 1
            neighbor.h = heuristic(neighbor, goal_node)
            neighbor.parent = current_node

            if neighbor in open_list:
                for node in open_list:
                    if node == neighbor and node.g > neighbor.g:
                        node.g = neighbor.g
                        node.parent = neighbor.parent
            else:
                heapq.heappush(open_list, neighbor)

        iteration += 1

    return None

def action(path):
    if not path or len(path) < 2:
        return 0
    dy, dx = path[1].x - path[0].x, path[1].y - path[0].y
    if dy == -1:
        return 0
    elif dx == 1:
        return 1
    elif dy == 1:
        return 2
    elif dx == -1:
        return 3

def find_next_move(map, start, goal, max_iter=100):
    start = (start // map.shape[1], start % map.shape[1])
    goal = (goal // map.shape[1], goal % map.shape[1])
    if disstance(start, goal) == 1:
        return action([Node(*start), Node(*goal)])
    path = astar(map, start, goal, max_iter)
    return action(path)

CARRYING = 6
PLAYER = 11
ENEMY = 12
NOTHING = 13
RESOURCE = 14
BASE = 15
BARRACK = 16
WORKER = 17
LIGHT = 18
HEAVY = 19
RANGED = 20


NOOP = 0
MOVE = 1
HARVEST = 2
RETURN = 3
PRODUCE = 4
ATTACK = 5

def get_closest_unit(pos:int, unit_list:list, width:int)->int:
    min_dis = 1000
    min_unit = -1
    pos_x = pos % width
    pos_y = pos // width
    for unit in unit_list:
        unit_x = unit % width
        unit_y = unit // width
        dis = abs(pos_x - unit_x) + abs(pos_y - unit_y)
        if dis < min_dis:
            min_dis = dis
            min_unit = unit
    return min_unit


class ScriptAgent:
    def __init__(self,w,init_resource=5):
        self.max_worker = 3
        self.max_harvest_worker = 2
        self.w = w

        self.resource_owned = init_resource
        
        self.allay_base_list = []
        self.allay_barracks_list = []
        self.allay_worker_list = []
        self.allay_light_list = []
        self.allay_heavy_list = []
        self.allay_ranged_list = []

        self.op_base_list = []
        self.op_barracks_list = []
        self.op_worker_list = []
        self.op_light_list = []
        self.op_heavy_list = []
        self.op_ranged_list = []

        self.resource_list = []
        self.avaliable_resource_list = []
    
    def reset(self,w):
        self.max_worker = 3
        self.max_harvest_worker = 2
        self.w = w

        self.resource_owned = 0
        
        self.allay_base_list = []
        self.allay_barracks_list = []
        self.allay_worker_list = []
        self.allay_light_list = []
        self.allay_heavy_list = []
        self.allay_ranged_list = []

        self.op_base_list = []
        self.op_barracks_list = []
        self.op_worker_list = []
        self.op_light_list = []
        self.op_heavy_list = []
        self.op_ranged_list = []

        self.resource_list = []
        self.avaliable_resource_list = []

    def get_action(self,state,unit,action_mask=None):
        self.allay_base_list = []
        self.allay_barracks_list = []
        self.allay_worker_list = []
        self.allay_light_list = []
        self.allay_heavy_list = []
        self.allay_ranged_list = []

        self.op_base_list = []
        self.op_barracks_list = []
        self.op_worker_list = []
        self.op_light_list = []
        self.op_heavy_list = []
        self.op_ranged_list = []

        self.resource_list = []
        self.avaliable_resource_list = []
        for y in range(state.shape[0]):
            for x in range(state.shape[1]):
                if state[y][x][BASE] == 1 and state[y][x][PLAYER]:
                    self.allay_base_list.append(y * state.shape[1] + x)
                elif state[y][x][BARRACK] == 1 and state[y][x][PLAYER]:
                    self.allay_barracks_list.append(y * state.shape[1] + x)
                elif state[y][x][WORKER] == 1 and state[y][x][PLAYER]:
                    self.allay_worker_list.append(y * state.shape[1] + x)
                elif state[y][x][LIGHT] == 1 and state[y][x][PLAYER]:
                    self.allay_light_list.append(y * state.shape[1] + x)
                elif state[y][x][HEAVY] == 1 and state[y][x][PLAYER]:
                    self.allay_heavy_list.append(y * state.shape[1] + x)
                elif state[y][x][RANGED] == 1 and state[y][x][PLAYER]:
                    self.allay_ranged_list.append(y * state.shape[1] + x)
                elif state[y][x][BASE] == 1 and state[y][x][ENEMY]:
                    self.op_base_list.append(y * state.shape[1] + x)
                elif state[y][x][BARRACK] == 1 and state[y][x][ENEMY]:
                    self.op_barracks_list.append(y * state.shape[1] + x)
                elif state[y][x][WORKER] == 1 and state[y][x][ENEMY]:
                    self.op_worker_list.append(y * state.shape[1] + x)
                elif state[y][x][LIGHT] == 1 and state[y][x][ENEMY]:
                    self.op_light_list.append(y * state.shape[1] + x)
                elif state[y][x][HEAVY] == 1 and state[y][x][ENEMY]:
                    self.op_heavy_list.append(y * state.shape[1] + x)
                elif state[y][x][RANGED] == 1 and state[y][x][ENEMY]:
                    self.op_ranged_list.append(y * state.shape[1] + x)
                elif state[y][x][RESOURCE] == 1:
                    self.resource_list.append(y * state.shape[1] + x)

        # if resource near base, resource is avaliable, range is w/2
        for resource in self.resource_list:
            for base in self.allay_base_list:
                if distance(resource, base, state.shape[1]) <= self.w // 2:
                    self.avaliable_resource_list.append(resource)
                    break
        
        unit_x = unit % self.w
        unit_y = unit // self.w

        # worker action
        if state[unit_y][unit_x][PLAYER] == 1 and state[unit_y][unit_x][WORKER] == 1:
            # if any enemy unit near worker, attack, range is 3
            for enemy in self.op_worker_list:
                if distance(unit, enemy, self.w) <= 3:
                    if disstance(unit,enemy,self.w) == 1:
                        p = np.array([3,3])+np.array([enemy // self.w - unit_y, enemy % self.w - unit_x])
                        u = 7*p[0]+p[1]
                        return [unit, ATTACK, 0, 0, 0, 0, 0, u]
                    else:
                        dir = find_next_move(state[:,:,PLAYER] + state[:,:,ENEMY] + state[:,:,RESOURCE], unit, enemy)
                        return [unit, MOVE, dir, dir, dir, dir, 0, 0]

            if len(self.avaliable_resource_list) > 0:
                harvest_workers = deque(maxlen=self.max_harvest_worker)
                target_resources = deque(maxlen=self.max_harvest_worker)
                disstance_list = deque(maxlen=self.max_harvest_worker)
                for worker in self.allay_worker_list:
                    for resource in self.avaliable_resource_list:
                        dis = distance(worker, resource, state.shape[1])
                        if len(disstance_list) < self.max_harvest_worker:
                            harvest_workers.append(worker)
                            target_resources.append(resource)
                            disstance_list.append(dis)
                        else:
                            if dis < max(disstance_list):
                                index = disstance_list.index(max(disstance_list))
                                harvest_workers[index] = worker
                                target_resources[index] = resource
                                disstance_list[index] = dis
                if unit in harvest_workers:
                    index = harvest_workers.index(unit)
                    if state[unit_y][unit_x][CARRYING] == 0:
                        target = target_resources[index]
                        dis = disstance_list[index]
                        obstacle = state[:,:,PLAYER] + state[:,:,ENEMY] + state[:,:,RESOURCE]
                        obstacle[target // state.shape[1]][target % state.shape[1]] = 0
                        obstacle[unit_y][unit_x] = 0
                        dir = find_next_move(obstacle, unit, target)
                        if dis == 1:
                            return [unit, HARVEST, dir, dir, dir, dir, 0, 0]
                        else:
                            return [unit, MOVE, dir, dir, dir, dir, 0, 0]
                    elif state[unit_y][unit_x][CARRYING] > 0:
                        nearest_base = get_closest_unit(unit, self.allay_base_list, self.w)
                        obstacle = state[:,:,PLAYER] + state[:,:,ENEMY] + state[:,:,RESOURCE]
                        obstacle[nearest_base // state.shape[1]][nearest_base % state.shape[1]] = 0
                        obstacle[unit_y][unit_x] = 0
                        dir = find_next_move(obstacle, unit, nearest_base)
                        if distance(unit, nearest_base, self.w) == 1:
                            self.resource_owned += 1
                            return [unit, RETURN, dir, dir, dir, dir, 0, 0]
                        else:
                            return [unit, MOVE, dir, dir, dir, dir, 0, 0]
                else:
                    #move to nearest enemy base or barracks or worker and attack
                    target_lst = self.op_base_list + self.op_barracks_list + self.op_worker_list
                    if len(target_lst) > 0:
                        target = get_closest_unit(unit, target_lst, self.w)
                        target_x = target % self.w
                        target_y = target // self.w
                        obstacle = state[:,:,PLAYER] + state[:,:,ENEMY] + state[:,:,RESOURCE]
                        obstacle[target_y][target_x] = 0
                        obstacle[unit_y][unit_x] = 0
                        dir = find_next_move(obstacle, unit, target)
                        if distance(unit, target, self.w) == 1:
                            p = np.array([3,3])+np.array([target_y-unit_y,target_x-unit_x])
                            u = 7*p[0]+p[1]
                            return [unit, ATTACK, dir, dir, dir, dir, 0, u]
                        else:
                            return [unit, MOVE, dir, dir, dir, dir, 0, 0]
        #base action
        elif unit in self.allay_base_list:
            if len(self.allay_worker_list) < self.max_harvest_worker:
                nearest_resource = get_closest_unit(unit, self.resource_list, self.w)
                obstacle = state[:,:,PLAYER] + state[:,:,ENEMY] + state[:,:,RESOURCE]
                obstacle[nearest_resource // state.shape[1]][nearest_resource % state.shape[1]] = 0
                obstacle[unit_y][unit_x] = 0
                dir = find_next_move(obstacle, unit, nearest_resource)
                if distance(unit, nearest_resource, self.w) == 1:
                    new_dir = (dir + 1)%4
                    if next_pos(unit_x, unit_y, new_dir, self.w) is not None:
                        next_pos_x, next_pos_y = next_pos(unit_x, unit_y, new_dir, self.w)
                        if state[next_pos_y][next_pos_x][PLAYER] == 0 and state[next_pos_y][next_pos_x][ENEMY] == 0 and state[next_pos_y][next_pos_x][RESOURCE] == 0:
                            self.resource_owned -= 1
                            return [unit, PRODUCE, new_dir, new_dir, new_dir, new_dir, 3, 0]
                    new_dir = (dir + 3)%4
                    if next_pos(unit_x, unit_y, new_dir, self.w) is not None:
                        next_pos_x, next_pos_y = next_pos(unit_x, unit_y, new_dir, self.w)
                        if state[next_pos_y][next_pos_x][PLAYER] == 0 and state[next_pos_y][next_pos_x][ENEMY] == 0 and state[next_pos_y][next_pos_x][RESOURCE] == 0:
                            self.resource_owned -= 1
                            return [unit, PRODUCE, new_dir, new_dir, new_dir, new_dir, 3, 0]
                    new_dir = (dir + 2)%4
                    if next_pos(unit_x, unit_y, new_dir, self.w) is not None:
                        next_pos_x, next_pos_y = next_pos(unit_x, unit_y, new_dir, self.w)
                        if state[next_pos_y][next_pos_x][PLAYER] == 0 and state[next_pos_y][next_pos_x][ENEMY] == 0 and state[next_pos_y][next_pos_x][RESOURCE] == 0:
                            self.resource_owned -= 1
                            return [unit, PRODUCE, new_dir, new_dir, new_dir, new_dir, 3, 0]
                    return [unit, NOOP, 0, 0, 0, 0, 0, 0]

                else:
                    self.resource_owned -= 1
                    return [unit, PRODUCE, dir, dir, dir, dir, 3, 0]
            else:
                nearest_enemy = get_closest_unit(unit, self.op_worker_list +self.op_barracks_list+self.op_base_list, self.w)
                obstacle = state[:,:,PLAYER] + state[:,:,ENEMY] + state[:,:,RESOURCE]
                obstacle[nearest_enemy // state.shape[1]][nearest_enemy % state.shape[1]] = 0
                obstacle[unit_y][unit_x] = 0
                dir = find_next_move(obstacle, unit, nearest_enemy)
                if distance(unit, nearest_enemy, self.w) == 1:
                    new_dir = (dir + 1)%4
                    if next_pos(unit_x, unit_y, new_dir, self.w) is not None:
                        next_pos_x, next_pos_y = next_pos(unit_x, unit_y, new_dir, self.w)
                        if state[next_pos_y][next_pos_x][PLAYER] == 0 and state[next_pos_y][next_pos_x][ENEMY] == 0 and state[next_pos_y][next_pos_x][RESOURCE] == 0:
                            self.resource_owned -= 1
                            return [unit, PRODUCE, new_dir, new_dir, new_dir, new_dir, 3, 0]
                    new_dir = (dir + 3)%4
                    if next_pos(unit_x, unit_y, new_dir, self.w) is not None:
                        next_pos_x, next_pos_y = next_pos(unit_x, unit_y, new_dir, self.w)
                        if state[next_pos_y][next_pos_x][PLAYER] == 0 and state[next_pos_y][next_pos_x][ENEMY] == 0 and state[next_pos_y][next_pos_x][RESOURCE] == 0:
                            self.resource_owned -= 1
                            return [unit, PRODUCE, new_dir, new_dir, new_dir, new_dir, 3, 0]
                    new_dir = (dir + 2)%4
                    if next_pos(unit_x, unit_y, new_dir, self.w) is not None:
                        next_pos_x, next_pos_y = next_pos(unit_x, unit_y, new_dir, self.w)
                        if state[next_pos_y][next_pos_x][PLAYER] == 0 and state[next_pos_y][next_pos_x][ENEMY] == 0 and state[next_pos_y][next_pos_x][RESOURCE] == 0:
                            self.resource_owned -= 1
                            return [unit, PRODUCE, new_dir, new_dir, new_dir, new_dir, 3, 0]
                    return [unit, NOOP, 0, 0, 0, 0, 0, 0]

                else:
                    self.resource_owned -= 1
                    return [unit, PRODUCE, dir, dir, dir, dir, 3, 0]
        #barracks action
        elif unit in self.allay_barracks_list:
            nearest_enemy = get_closest_unit(unit, self.op_worker_list +self.op_barracks_list+self.op_base_list, self.w)
            obstacle = state[:,:,PLAYER] + state[:,:,ENEMY] + state[:,:,RESOURCE]
            obstacle[nearest_enemy // state.shape[1]][nearest_enemy % state.shape[1]] = 0
            obstacle[unit_y][unit_x] = 0
            dir = find_next_move(obstacle, unit, nearest_enemy)
            if distance(unit, nearest_enemy, self.w) == 1:
                new_dir = (dir + 1)%4
                if next_pos(unit_x, unit_y, new_dir, self.w) is not None:
                    next_pos_x, next_pos_y = next_pos(unit_x, unit_y, new_dir, self.w)
                    if state[next_pos_y][next_pos_x][PLAYER] == 0 and state[next_pos_y][next_pos_x][ENEMY] == 0 and state[next_pos_y][next_pos_x][RESOURCE] == 0:
                        self.resource_owned -= 1
                        return [unit, PRODUCE, new_dir, new_dir, new_dir, new_dir, 4, 0]
                new_dir = (dir + 3)%4
                if next_pos(unit_x, unit_y, new_dir, self.w) is not None:
                    next_pos_x, next_pos_y = next_pos(unit_x, unit_y, new_dir, self.w)
                    if state[next_pos_y][next_pos_x][PLAYER] == 0 and state[next_pos_y][next_pos_x][ENEMY] == 0 and state[next_pos_y][next_pos_x][RESOURCE] == 0:
                        self.resource_owned -= 1
                        return [unit, PRODUCE, new_dir, new_dir, new_dir, new_dir, 4, 0]
                new_dir = (dir + 2)%4
                if next_pos(unit_x, unit_y, new_dir, self.w) is not None:
                    next_pos_x, next_pos_y = next_pos(unit_x, unit_y, new_dir, self.w)
                    if state[next_pos_y][next_pos_x][PLAYER] == 0 and state[next_pos_y][next_pos_x][ENEMY] == 0 and state[next_pos_y][next_pos_x][RESOURCE] == 0:
                        self.resource_owned -= 1
                        return [unit, PRODUCE, new_dir, new_dir, new_dir, new_dir, 4, 0]
                return [unit, NOOP, 0, 0, 0, 0, 0, 0]

            else:
                self.resource_owned -= 1
                return [unit, PRODUCE, dir, dir, dir, dir, 4, 0]

        else:
            return [unit, NOOP, 0, 0, 0, 0, 0, 0]

if __name__ == '__main__':
    seed = 0

    num_envs = 1

    env = MicroRTSVecEnv(
            num_envs=num_envs,
            max_steps=5000,
            ai2s=[microrts_ai.randomAI for _ in range(num_envs)],
            map_path='maps/10x10/basesWorkers10x10.xml',
            reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
        )
    next_obs = env.reset()

    outcomes = []

    resource1 = 0

    sa = ScriptAgent(10)
    for games in range(100):
        for step in range(5000):
            env.render()
            unit_masks = np.array(env.vec_client.getUnitLocationMasks()).reshape(num_envs, -1)
            actions = []
            units = []
            for unit_mask in unit_masks:
                if np.sum(unit_mask) == 0:
                    units.append(-1)
                else:
                    units.append(np.random.choice(np.where(unit_mask == 1)[0]))
            for i in range(len(units)):
                if units[i] == -1:
                    actions.append([0, 0, 0, 0, 0, 0, 0, 0])
                else:
                    action_mask_list = np.array(env.vec_client.getUnitActionMasks(np.array(units))).reshape(len(units), -1)
                    actions.append(sa.get_action(next_obs[0], units[-1], action_mask_list[0]))
            next_obs, rs, ds, infos = env.step(actions)
            time.sleep(0.01)
