import numpy as np
import heapq

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

def astar(map, start, end):
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    open_list = []
    closed_list = []

    heapq.heappush(open_list, (start_node.f, id(start_node), start_node))

    while len(open_list) > 0:
        current_node = heapq.heappop(open_list)[2]

        closed_list.append(current_node)

        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]

        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            if node_position[0] > (len(map) - 1) or node_position[0] < 0 or node_position[1] > (len(map[len(map)-1]) -1) or node_position[1] < 0:
                continue

            if map[node_position[0]][node_position[1]] != 0:
                continue

            new_node = Node(current_node, node_position)

            children.append(new_node)

        for child in children:

            if child in closed_list:
                continue

            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            if len([i for i in open_list if child == i[2] and child.g > i[2].g]) > 0:
                continue

            heapq.heappush(open_list, (child.f, id(child), child))

    return None

def get_next_move(maze, start, end):
    if maze[start[0]][start[1]] == 1:
        maze[start[0]][start[1]] = 0
    if maze[end[0]][end[1]] == 1:
        maze[end[0]][end[1]] = 0
    path = astar(maze, start, end)
    if path is None or len(path) < 2:
        return 0
    dy, dx = path[1][0] - path[0][0], path[1][1] - path[0][1]
    if dy == -1:
        return 0
    elif dx == 1:
        return 1
    elif dy == 1:
        return 2
    elif dx == -1:
        return 3


maze = np.array([
    [1, 0, 0, 1, 0],
    [1, 1, 0, 1, 1],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1],
    [0, 0, 0, 0, 1]
])

start = (0, 2)
end = (4, 4)

next_move = get_next_move(maze, start, end)
print(next_move)

