import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

l = []
with open('input.txt', 'r') as file:
    line = file.readline()

    while line:
      l.append(line.strip())
      line = file.readline()

adj = l[3:]

start,goal = tuple(map(int,l[1:3]))

graph = {}
adj2 = []

for i in adj:

  adj2.append(tuple(i.split()))

map_graph = {}
for i in adj2:
  v,j,w = i

  if v in map_graph:
    map_graph[v].append((w,j))
    continue

  map_graph[v] = [(w,j)]

def convert_graph(graph):
  d_graph = {}
  for node, neighbor in graph.items():
    d_graph[int(node)] = [(float(weight) , int(target)) for weight,target in neighbor]
    
  return d_graph

d = convert_graph(map_graph)


def dijkstra_algorithm(start, goal, graph):
    minimum_cost = {node: float("inf") for node in graph}
    minimum_cost[start] = 0 
    
    path = {node: None for node in graph}
    
    visited_nodes = []
    visited_edges = []
    
    unvisited = list(graph.keys())

    while unvisited:
        current_node = None
        current_cost = float("inf")
        for node in unvisited:
            if minimum_cost[node] < current_cost:
                current_cost = minimum_cost[node]
                current_node = node

        if current_node is None:
            break
        
       
        visited_nodes.append(current_node)
        unvisited.remove(current_node)

        
        if current_node == goal:
            break
        
        
        for cost, neighbor in graph[current_node]:
            dist = current_cost + float(cost)
            if dist < minimum_cost[neighbor]:
                minimum_cost[neighbor] = dist
                path[neighbor] = current_node
                visited_edges.append((current_node, neighbor))
    
    
    path_shortest = []
    current = goal
    while current is not None:
        path_shortest.append(current)
        current = path[current]
    path_shortest = path_shortest[::-1]

    minimum_cost_res = []

    for i,k in minimum_cost.items():
      if i in path_shortest:
        minimum_cost_res.append(k)



    return minimum_cost_res, path_shortest, visited_nodes, visited_edges



mc,sp,vn,ve = dijkstra_algorithm(start, goal , d)

string_shortest_path = " ".join(str(item) for item in sp)
min_cost_string = " ".join(str(item) for item in mc)

file_name = "output.txt"

directory = os.path.dirname(__file__)

file_path = os.path.join(directory,file_name)

with open(file_path, "w") as file:
   file.write(string_shortest_path + '\n' + min_cost_string)

file.close()
   

coords = []
with open("coords.txt") as file:
  
  for line in file:
    coords.append(tuple((line.strip().split())))

floating_coords = []
for item in coords:
  (x,y) = item 
  x = float(x)
  y = float(y)
  floating_coords.append((x,y))


fig, ax = plt.subplots(figsize=(10,8))
ax.set_xlim(min(x for x, y in floating_coords) - 1, max(x for x, y in floating_coords) + 1)
ax.set_ylim(min(y for x, y in floating_coords) - 1, max(y for x, y in floating_coords) + 1)

x, y = zip(*floating_coords)
scatter = ax.scatter(x, y, color='blue', s=25, zorder=5)

def draw_edge():
  for i, neighbors in d.items():
    for weight, j in neighbors:
          xi, yi = floating_coords[i - 1] 
          xj, yj = floating_coords[j - 1] 
          ax.plot([xi, xj], [yi, yj], color='gray', linestyle='--', zorder=1)


labels = []
for i in d.keys():
    labels.append(i)
    for i, label in enumerate(labels):
      ax.annotate(label, (x[i], y[i]), textcoords="offset points", xytext=(20,0), ha='right')


def draw_final_path():
    if len(sp) > 1:
      for i in range(len(sp) - 1):
        x1, y1 = floating_coords[sp[i] - 1]
        x2, y2 = floating_coords[sp[i + 1] - 1]
        ax.plot([x1, x2], [y1, y2], color='green', linewidth=2, zorder=4)
        ax.scatter([x1, x2], [y1, y2], color='green', s=100, zorder = 7)


def update(frame):

      
      ax.set_xlim(min(x for x, y in floating_coords) - 1, max(x for x, y in floating_coords) + 1)
      ax.set_ylim(min(y for x, y in floating_coords) - 1, max(y for x, y in floating_coords) + 1)

      scatter = ax.scatter(x, y, color='blue', s=50, zorder=5)
      draw_edge()

      if frame < len(vn):
        current_node = vn[frame]
        ax.scatter(floating_coords[current_node - 1][0], floating_coords[current_node - 1][1], color='red', s=100, zorder=6)
        for edge in ve:
            if edge[0] == current_node or edge[1] == current_node:
                x1, y1 = floating_coords[edge[0] - 1]
                x2, y2 = floating_coords[edge[1] - 1]
                ax.plot([x1, x2], [y1, y2], color='red', linewidth=2, zorder=3)


      if frame == len(vn):
          draw_final_path()

      return scatter


ani = animation.FuncAnimation(fig, update, frames=len(vn) + 1, repeat=False, interval=1000)

ani.save('DijkstraSim.mp4', writer='ffmpeg')


