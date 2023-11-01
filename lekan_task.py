import csv
import random
import math
import pandas as pd
#Step 1: Read the data from CSV files
#Step 2: Define the Genetic Algorithm and Simulated Annealing components
#
#    Genetic Algorithm (GA):
#        Representation: Each chromosome is a mapping of courses to time slots and venues.
#        Selection: Randomly select chromosomes based on their fitness.#
#        Crossover: Combine two chromosomes to form a new chromosome.
#        Mutation: Randomly change a time slot or venue for a course.

#    Simulated Annealing (SA):
#        Start with an initial solution.
#        At each step, perturb the current solution to get a neighboring solution.
#        If the new solution is better, accept it. If it's worse, accept it with a certain probability.
#        Reduce the temperature at each step.

#Step 3: Define the Fitness Function

#A good fitness function should consider:

#    No student takes two exams on the same day.
#    Exam venue capacity >= course population.
#    The time slot is within the allowed range (9am - 5pm).
def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)[1:]  # Exclude header
    return data

venues = read_csv("venue.csv")
courses = read_csv("courses.csv")
# day = read_csv("day.csv")

#    Initialize Population: A random set of schedules.
#   Parent Selection: Using a method like tournament selection.
#    Crossover: Combine two chromosomes to form a new chromosome.
#    Mutation: Randomly change a time slot or venue for a course.
#
#
#
# Constants
POPULATION_SIZE = 100
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.1
MAX_GENERATIONS = 1000
SOME_THRESHOLD = 0.99  # Some arbitrary fitness threshold to decide convergence

# Sample course and venue data (you'd likely replace this with data from your CSV files)
#courses = [("Course1", "C1", 30), ("Course2", "C2", 25), ...]
#venues = [("Venue1", 50), ("Venue2", 60), ...]


# Initialize a random population
def initialize_population():
    return [random_chromosome() for _ in range(POPULATION_SIZE)]


# Generate a random chromosome
def random_chromosome():
    return [{'course': random.choice(courses), 'venue': random.choice(venues),'day': random_day(), 'time': random_time_slot()} for _ in courses]


# Generate a random time slot within 9am-5pm
def random_time_slot():
    start = random.randint(9, 16)
    end = start + 1
    return f"{start}:00-{end}:00"

def random_day():
    return str(random.choice(["Monday","Tuesday","Wednesday","Thursday","Friday"]))
#    No student should take two exams in the same day.
#    Exam venue capacity should be greater than or equal to course population.
#    The time slot should be within the allowed range (9am - 5pm).

#For this purpose, let's assign penalty points for every constraint violation. The fewer the penalty points, the higher the fitness.
# Fitness function to evaluate a chromosome
def fitness(chromosome):
    penalty = 0
    daily_courses = {}
    print("Process Initiated .....")
    for gene in chromosome:

        print("Gene per chromosome")
        print (gene)
        sn, course_code, course_population = gene['course']
        sn, venue_name, venue_capacity = gene['venue']
        time_slot = gene['time']
        day_slot = gene['day']

        if venue_capacity < course_population:
            penalty += abs((int(course_population) - int(venue_capacity)))

        start_time, end_time = [int(t.split(':')[0]) for t in time_slot.split('-')]
        if start_time < 9 or end_time > 17:
            penalty += 50
        print("Daily Courses")
        print(daily_courses)

        if gene['day'] not in daily_courses:
            daily_courses[gene['day']] = set()
        if course_code in daily_courses[gene['day']]:
            penalty += 100
        else:
            daily_courses[gene['day']].add(course_code)

    print(penalty)

    return 1 / (1 + penalty)


# Tournament selection for genetic algorithm
def tournament_selection(population):
    candidates = random.sample(population, 5)
    return max(candidates, key=fitness)


# One-point crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1)-1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


# Mutation function
def mutate(chromosome):
    mutation_point = random.randint(0, len(chromosome)-1)
    chromosome[mutation_point] = {'course': random.choice(courses), 'venue': random.choice(venues), 'time': random_time_slot(), 'day': random_day()}
    return chromosome


# Genetic algorithm
def genetic_algorithm():
    population = initialize_population()

    for generation in range(MAX_GENERATIONS):
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            if random.random() < CROSSOVER_RATE:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2

            if random.random() < MUTATION_RATE:
                child1 = mutate(child1)
            if random.random() < MUTATION_RATE:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population

        best_chromosome = max(population, key=fitness)

        if fitness(best_chromosome) > SOME_THRESHOLD:
            break

    return best_chromosome


# Simulated annealing
#Simulated Annealing (SA) is an optimization technique inspired by the annealing process in metallurgy. The basic idea 
#is to start with a high "temperature" and gradually "cool" the system. At high temperatures,
#the algorithm is more likely to accept worse solutions, whereas at lower temperatures, it becomes more selective.
def simulated_annealing():
    current_solution = random_chromosome()
    current_fitness = fitness(current_solution)

    temperature = 1.0
    cooling_rate = 0.995
    min_temperature = 0.001

    while temperature > min_temperature:
        neighbor = generate_neighbor(current_solution)
        neighbor_fitness = fitness(neighbor)
        delta_fitness = neighbor_fitness - current_fitness

        if delta_fitness > 0:
            current_solution = neighbor
            current_fitness = neighbor_fitness
        else:
            probability = math.exp(delta_fitness / temperature)
            if random.random() < probability:
                current_solution = neighbor
                current_fitness = neighbor_fitness

        temperature *= cooling_rate

    return current_solution


# Generate a neighboring solution for simulated annealing
def generate_neighbor(solution):
    return mutate(list(solution))


# Write data to a CSV file
def write_to_csv(data, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)


if __name__ == "__main__":
    # For the sake of illustration, let's assume we're using the genetic algorithm to compute the best timetable
    best_timetable = genetic_algorithm()
    # Convert best_timetable into the desired format for CSV if needed
    # Write the best timetable to a CSV file
    print(best_timetable)

    df = pd.DataFrame(best_timetable)
    print(df.head())
    df.to_excel("timetable_outputs.xlsx")
    # write_to_csv(best_timetable, "timetable_output.csv")
    write_to_csv(df, "timetable_output.csv")
    print("Process Completed Successfully")
