import numpy as np

class NaivePathFinder:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def get_obstacles(self, grid_state)->set:
        obstacles = set()
        for i in range(self.height):
            for j in range(self.width):
                if grid_state[i][j][10] == 1 or grid_state[i][j][11] == 1 and grid_state[i][j][12] == 1:
                    obstacles.add(i*self.width + j)
        return obstacles
    
    def distance(self, current:int, goal:int)->int:
        return abs(current // self.width - goal // self.width) + abs(current % self.width - goal % self.width)
    
    def get_neighbors(self, current:int)->list:
        neighbors = []
        if current - self.width >= 0:
            neighbors.append(current - self.width)
        if current + 1 < self.width * self.height:
            neighbors.append(current + 1)
        if current + self.width < self.width * self.height:
            neighbors.append(current + self.width)
        if current - 1 >= 0:
            neighbors.append(current - 1)
        return neighbors
    
    def find_path(self, start:int, goal:int, grid_state:int)->list:
        obstacles = self.get_obstacles(grid_state)
        queue = [start]
        visited = set()
        parent = {}
        while queue:
            current = queue.pop(0)
            if current == goal:
                break
            visited.add(current)
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited and neighbor not in queue and neighbor not in obstacles:
                    queue.append(neighbor)
                    parent[neighbor] = current
        path = [goal]
        while path[-1] != start:
            path.append(parent[path[-1]])
        path.reverse()
        return path
    
    def next_pos(self, gird, current_pos, goal_pos, max_range):
        if self.distance(current_pos, goal_pos) < max_range:
            path = self.find_path(current_pos, goal_pos, gird)
            return path[0]
        else:
            next_pos = current_pos
            min_distance = self.distance(current_pos, goal_pos)
            for neighbor in self.get_neighbors(current_pos):
                if self.distance(neighbor, goal_pos) < min_distance:
                    next_pos = neighbor
                    min_distance = self.distance(neighbor, goal_pos)
            return next_pos
        
    def get_action(self, grid, current_pos, goal_pos, max_range):
        next_pos = self.next_pos(grid, current_pos, goal_pos, max_range)
        if next_pos == current_pos - self.width:
            return 0
        elif next_pos == current_pos + 1:
            return 1
        elif next_pos == current_pos + self.width:
            return 2
        elif next_pos == current_pos - 1:
            return 3
        else:
            return 4
        
