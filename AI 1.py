import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

N = 100                  
cities = np.random.rand(N,2)*10

# Distance
def distance(route):
    total = 0

    for i in range(len(route)):
        a = cities[route[i]]
        b = cities[route[(i+1)%len(route)]]

        total += np.linalg.norm(a-b)

    return total

# Plot route
def plot_route(route,title):
    plt.figure(figsize=(6,6))
    x = cities[route,0]
    y = cities[route,1]
    plt.plot(np.append(x,x[0]),
             np.append(y,y[0]),
             'r--',
             linewidth=1)
    plt.scatter(cities[:,0],cities[:,1],
                facecolors='none',
                edgecolors='r')
    plt.title(title)
    plt.xlim(0,10)
    plt.ylim(0,10)
    plt.show()


# Initial route
route = np.arange(N)
np.random.shuffle(route)

print("Initial distance:",distance(route))

plot_route(route,"Initial Route")


# 2-opt optimization
def two_opt(route):

    best = route.copy()
    improved = True

    while improved:
        improved = False
        best_distance = distance(best)
        for i in range(1,len(best)-2):
            for j in range(i+1,len(best)):
                if j-i == 1:
                    continue
                new_route = best.copy()
                # Reverse segment
                new_route[i:j] = best[j-1:i-1:-1]

                new_distance = distance(new_route)

                if new_distance < best_distance:
                    best = new_route
                    best_distance = new_distance
                    improved = True
    return best

# Optimization
best_route = two_opt(route)
print("Optimized distance:",distance(best_route))
plot_route(best_route,"Optimized Route")