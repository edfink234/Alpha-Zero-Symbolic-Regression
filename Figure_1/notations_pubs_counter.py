from scholarly import scholarly
import os

# Function to search Google Scholar and count publications over time
def count_publications(queries):
    publication_years = []
    publication_titles = []
    for query in queries:
        search_query = scholarly.search_pubs(query)
        for publication in search_query:
            try:
                if publication['bib'].get('title') not in publication_titles:
                    publication_titles.append(publication['bib'].get('title'))
                    publication_years.append(int(publication['bib'].get('pub_year')))
            except:
                continue
        
    return publication_years

# Defining the search query
postfix_queries = ('"postfix notation" AND "symbolic regression"', '"reverse polish notation" AND "symbolic regression"', '"postfix expression" AND "symbolic regression"')
prefix_queries = ('"prefix notation" AND "symbolic regression"', '"polish notation" AND "symbolic regression"', '"prefix expression" AND "symbolic regression"')
acyclic_graph_queries = ('"acyclic graph" AND "symbolic regression"', '"acyclic-graph" AND "symbolic regression"')

# Fetching publication years
#publication_years_postfix = count_publications(postfix_queries)
#with open("postfix_pub_years.txt", "w") as file:
#    for i in publication_years_postfix:
#        file.write(f"{i}\n")

#publication_years_prefix = count_publications(prefix_queries)
#with open("prefix_pub_years.txt", "w") as file:
#    for i in publication_years_prefix:
#        file.write(f"{i}\n")
#
#publication_years_acyclic_graph = count_publications(acyclic_graph_queries)
#with open("acyclic_graph_pub_years.txt", "w") as file:
#    for i in publication_years_acyclic_graph:
#        file.write(f"{i}\n")

#Visualize:
import matplotlib.pyplot as plt

# Read integers from the file
file_path = 'prefix_pub_years.txt'
with open(file_path, 'r') as file:
    integers_prefix = [int(line.strip()) for line in file.readlines()]
file_path = 'postfix_pub_years.txt'
with open(file_path, 'r') as file:
    integers_postfix = [int(line.strip()) for line in file.readlines()]
file_path = 'acyclic_graph_pub_years.txt'
with open(file_path, 'r') as file:
    integers_acyclic_graph = [int(line.strip()) for line in file.readlines()]

# Plot histogram
plt.hist(integers_acyclic_graph, bins=10, color='green', edgecolor='black', label = "acyclic-graph", alpha = 0.7)   # Adjust the number of bins as needed
plt.hist(integers_prefix, bins=10, color='skyblue', edgecolor='black', label = "prefix", alpha = 0.7)  # Adjust the number of bins as needed
plt.hist(integers_postfix, bins=10, color='orange', edgecolor='black', label = "postfix", alpha = 0.7)  # Adjust the number of bins as needed
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.title('# of Publication Mentions')
plt.grid(True)
plt.legend()
plt.savefig("pub_freqs.svg")
os.system(f"rsvg-convert -f pdf -o pub_freqs.pdf pub_freqs.svg")
os.system(f"rm pub_freqs.svg")



