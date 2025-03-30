import heapq

class IntelligentTransport:
  def __init__(self):
    self.graph = {}
    self.heuristics = {}

  def add_connection(self, origin, destination, cost):
    if origin not in self.graph:
      self.graph[origin] = []
    
    if destination not in self.graph:
      self.graph[destination] = []
    
    self.graph[origin].append((destination, cost))
    self.graph[destination].append((origin, cost))

  def define_heuristics(self, heuristics):
    self.heuristics = heuristics

  def best_route(self, start, destination):
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {node: float('inf') for node in self.graph}
    g_score[start] = 0
    f_score = {node: float('inf') for node in self.graph}
    f_score[start] = self.heuristics.get(start, 0)

    while open_set:
      _, current = heapq.heappop(open_set)

      if current == destination:
        return self.rebuild_route(came_from, current, start)
      
      for neigbor, cost in self.graph[current]:
        tentative_g_score = g_score[current] + cost

        if tentative_g_score < g_score[neigbor]:
          came_from[neigbor] = current
          g_score[neigbor] = tentative_g_score
          f_score[neigbor] = g_score[neigbor] + self.heuristics.get(neigbor, 0)
          heapq.heappush(open_set, (f_score[neigbor], neigbor))
        
    return None # No se encontró la mejor ruta
  
  def rebuild_route(self, came_from, current, start):
    route = []
    while current in came_from:
      route.append(current)
      current = came_from[current]

    if current == start:
      return list(reversed(route))
    return None

# Definir el sistema
system = IntelligentTransport()

# Agregar conexiones entre estaciones 
system.add_connection('A', 'B', 2)
system.add_connection('B', 'C', 3)
system.add_connection('A', 'D', 1)
system.add_connection('D', 'E', 4)
system.add_connection('E', 'C', 1)
system.add_connection('B', 'E', 2)
system.add_connection('A', 'F', 5)

# Definir heurística basada en la distancia estimada al destino
heuristics = { 'A': 5, 'B': 3, 'C': 0, 'D': 4, 'E': 2}
system.define_heuristics(heuristics)

# Buscar la mejor ruta de A a E
route = system.best_route('A', 'E')
print('Mejor ruta encontrada: ', route)