import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

class EpidemicGameSimulator:
    """
    An implementation of the evolutionary epidemic game model from Zhang et al. (2013),
    "Braess's Paradox in Epidemic Game".
    
    This class simulates the co-evolution of an epidemic and individual protective
    strategies on a network.
    """

    def __init__(self, G, delta, lambd, mu, b, c, kappa, I0):
        """
        Initializes the simulator with a graph and model parameters.
        
        Args:
            G (nx.Graph): The contact network.
            delta (float): The success rate of self-protection (δ in the paper).
            lambd (float): The transmission probability (λ in the paper).
            mu (float): The recovery probability (μ in the paper).
            b (float): The cost of self-protection.
            c (float): The cost of vaccination.
            kappa (float): The strength of selection for strategy imitation (κ).
            I0 (int): The initial number of infected individuals.
        """
        self.G = G
        self.N = len(G.nodes())
        self.nodes = list(G.nodes())
        self.adj = G.adj

        # Model Parameters
        self.delta = delta
        self.lambd = lambd
        self.mu = mu
        self.b = b
        self.c = c
        self.kappa = kappa
        self.I0 = I0

        # Node Attributes
        # Initialize strategies randomly
        self.strategies = {node: random.choice(['V', 'S', 'L']) for node in self.nodes}
        self.statuses = {node: 'S' for node in self.nodes} # S=Susceptible (for SIR), not self-protecting
        self.payoffs = {node: 0.0 for node in self.nodes}

    def _run_sir_spreading(self):
        """
        Runs one full SIR epidemic season on the network.
        """
        # 1. Determine who is susceptible at the start of the season
        susceptible_nodes = set()
        for node in self.nodes:
            if self.strategies[node] == 'L': # Laissez-faire is always susceptible
                susceptible_nodes.add(node)
            elif self.strategies[node] == 'S': # Self-protecting is sometimes susceptible
                if random.random() > self.delta: # Fails with probability 1-delta
                    susceptible_nodes.add(node)
        
        if not susceptible_nodes:
            return 0 # No one can get sick

        # 2. Seed the epidemic
        initial_infected = random.sample(list(susceptible_nodes), min(self.I0, len(susceptible_nodes)))
        
        # Reset statuses
        self.statuses = {node: 'S' for node in self.nodes}
        infected_set = set()
        for node in initial_infected:
            self.statuses[node] = 'I'
            infected_set.add(node)
        
        recovered_count = 0

        # 3. Run the spreading process until no one is infected
        while infected_set:
            newly_infected = set()
            
            # Check for new infections
            for infected_node in list(infected_set):
                for neighbor in self.adj[infected_node]:
                    # Check if neighbor is susceptible AND is in the susceptible pool for this season
                    if self.statuses[neighbor] == 'S' and neighbor in susceptible_nodes:
                        if random.random() < self.lambd:
                            newly_infected.add(neighbor)
            
            # Check for recoveries
            for infected_node in list(infected_set):
                if random.random() < self.mu:
                    self.statuses[infected_node] = 'R'
                    recovered_count += 1
                    infected_set.remove(infected_node)

            # Update statuses of newly infected
            for node in newly_infected:
                self.statuses[node] = 'I'
            infected_set.update(newly_infected)

        return recovered_count

    def _calculate_payoffs(self):
        """
        Calculates the payoff for each individual based on their strategy
        and final health status from the SIR season.
        """
        for node in self.nodes:
            strategy = self.strategies[node]
            status = self.statuses[node]
            
            payoff = 0
            if strategy == 'V':
                payoff = -self.c
            elif strategy == 'S':
                payoff = -self.b
                if status == 'R': # Got infected despite self-protection
                    payoff -= 1
            elif strategy == 'L':
                if status == 'R': # Got infected
                    payoff = -1
            
            self.payoffs[node] = payoff

    def _update_strategies(self):
        """
        Each individual updates their strategy for the next season by imitating
        a random neighbor based on the Fermi rule.
        """
        next_strategies = self.strategies.copy()
        for node in self.nodes:
            # Select a random neighbor; if isolated, keep current strategy
            neighbors = list(self.adj[node])
            if not neighbors:
                continue
            neighbor = random.choice(neighbors)
            
            # Get payoffs and neighbor's strategy
            payoff_i = self.payoffs[node]
            payoff_j = self.payoffs[neighbor]
            strategy_j = self.strategies[neighbor]
            
            # Fermi rule: probability of adopting neighbor's strategy
            imitation_prob = 1 / (1 + np.exp(-self.kappa * (payoff_j - payoff_i)))
            
            if random.random() < imitation_prob:
                next_strategies[node] = strategy_j
        
        self.strategies = next_strategies

    def run_simulation(self, seasons, burn_in):
        """
        Runs the full co-evolutionary simulation for a number of seasons.
        
        Args:
            seasons (int): Total number of seasons to run.
            burn_in (int): Number of initial seasons to discard before averaging results.
            
        Returns:
            dict: A dictionary with the final averaged epidemic size and strategy fractions.
        """
        history = []

        for season in range(seasons):
            # One full cycle of the co-evolutionary process
            self._run_sir_spreading()
            self._calculate_payoffs()
            self._update_strategies()
            
            # Record statistics after the burn-in period
            if season >= burn_in:
                R = sum(1 for s in self.statuses.values() if s == 'R') / self.N
                pV = sum(1 for s in self.strategies.values() if s == 'V') / self.N
                pS = sum(1 for s in self.strategies.values() if s == 'S') / self.N
                pL = sum(1 for s in self.strategies.values() if s == 'L') / self.N
                history.append({'R': R, 'pV': pV, 'pS': pS, 'pL': pL})

        # Average the results from the post-burn-in history
        avg_results = {key: np.mean([s[key] for s in history]) for key in history[0]}
        return avg_results

### Main Script to Replicate Figure 1(a)

if __name__ == "__main__":

  # --- Parameters from Figure 1 caption ---
  N_side = 50  # Creates a 50x50 = 2500 node grid
  G = nx.grid_2d_graph(N_side, N_side, periodic=True) # Square lattice with periodic boundaries
  lambd = 0.5  # Transmission probability
  mu = 0.3     # Recovery probability
  b = 0.1      # Cost of self-protection
  c = 0.4      # Cost of vaccination
  kappa = 10   # Selection strength
  I0 = 5       # Initial infected

  # --- Simulation Settings ---
  # Note: The paper averages over 100 runs and 1000+ seasons.
  # This is a simplified version for demonstration.
  total_seasons = 200
  burn_in_seasons = 100
  delta_values = np.linspace(0, 1.0, 21) # Range of self-protection effectiveness

  results = []

  print("Running simulations for a range of delta values...")
  for i, delta in enumerate(delta_values):
      print(f"  Simulating delta = {delta:.2f} ({i+1}/{len(delta_values)})...")
      simulator = EpidemicGameSimulator(G, delta, lambd, mu, b, c, kappa, I0)
      avg_result = simulator.run_simulation(seasons=total_seasons, burn_in=burn_in_seasons)
      results.append(avg_result)
  print("Simulations complete.")

  # --- Plotting the Results ---
  R = [r['R'] for r in results]
  pV = [r['pV'] for r in results]
  pS = [r['pS'] for r in results]
  pL = [r['pL'] for r in results]

  plt.figure(figsize=(10, 6))
  plt.plot(delta_values, R, 'o-', label=r'$R$ (Epidemic Size)', color='black')
  plt.plot(delta_values, pV, 's-', label=r'$p_V$ (Vaccinated)', color='green')
  plt.plot(delta_values, pS, '^-', label=r'$p_S$ (Self-Protected)', color='red')
  plt.plot(delta_values, pL, 'd-', label=r'$p_L$ (Laissez-faire)', color='blue')

  # Demarcate the paradoxical region
  plt.axvline(x=0.28, color='gray', linestyle='--')
  plt.axvline(x=0.42, color='gray', linestyle='--')

  plt.xlabel(r'Successful rate of self-protection ($\delta$)', fontsize=14)
  plt.ylabel('Fractions / Epidemic Size', fontsize=14)
  plt.title("Replication of Zhang et al. (2013) - Figure 1(a)", fontsize=16)
  plt.legend(fontsize=12)
  plt.grid(True, linestyle=':', alpha=0.6)
  plt.ylim(0, 1.0)
  plt.xlim(0, 1.0)
  plt.show()