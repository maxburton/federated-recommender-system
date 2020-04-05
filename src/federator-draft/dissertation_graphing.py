import matplotlib.pyplot as plt
import numpy as np

a = [1994,29,
1995,37,
1996,44,
1997,99,
1998,179,
1999,323,
2000,430,
2001,648,
2002,832,
2003,1170,
2004,1530,
2005,1890,
2006,2360,
2007,2830,
2008,3500,
2009,4330,
2010,5330,
2011,5970,
2012,7030,
2013,8410,
2014,9570,
2015,10900,
2016,11900,
2017,13400,
2018,15400]

year = []
occurrences = []

for i in range(len(a)):
    if i % 2 == 0:
        year.append(a[i])
    else:
        occurrences.append(a[i])

plt.title("Research in Recommender Systems Over Time")
plt.xlabel("Year")
plt.ylabel("Occurences")
plt.bar(year, occurrences)
plt.savefig("popularity_over_time.pdf", format="pdf")
plt.show()
