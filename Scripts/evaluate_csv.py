import pandas as pd

df1 = pd.read_csv('lda_tuning_results.csv')

a = df1[['Alpha', 'Beta', 'Coherence']].groupby(['Alpha', 'Beta']).max()

print('(Alfa, ', 'Beta)', 'Coherence')
print(a.index[a['Coherence'].argmax()],max(a['Coherence']))

df1 = df1.loc[((df1['Alpha'] == a.index[a['Coherence'].argmax()][0]) & (df1['Beta'] == a.index[a['Coherence'].argmax()][1]))].reset_index(drop = True)
df2 = pd.read_csv('nmf_tuning_results.csv')

k_values = df2['Topics']
coherences = df1['Coherence']
coherences2 = df2['Coherence']

#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.style.use("ggplot")
matplotlib.rcParams.update({"font.size": 14})

fig = plt.figure(figsize=(13,7))
# create the line plot
ax2 = plt.plot( k_values, coherences2, label='NMF' )
ax = plt.plot( k_values, coherences, label='LDA' )
plt.legend(['NMF', 'LDA'])
#ax.legend()
plt.xticks(k_values)
plt.xlabel("Number of Topics")
plt.ylabel("Mean Coherence")
# add the points
plt.scatter( k_values, coherences2, s=120)
plt.scatter( k_values, coherences, s=120)
# find and annotate the maximum point on the plot
ymax = max([max(coherences), max(coherences2)])
if max(coherences) > max(coherences2):
    xpos = coherences.argmax()#coherences.index(ymax)
else:
    xpos = coherences2.argmax()
best_k = k_values[xpos]
plt.annotate( "k=%d" % best_k, xy=(best_k, ymax), xytext=(best_k, ymax), textcoords="offset points", fontsize=16)
# show the plot
#plt.show()
plt.savefig('Grafico.jpg')


#####################################################################


plt.style.use("ggplot")
matplotlib.rcParams.update({"font.size": 14})

fig = plt.figure(figsize=(13,7))
# create the line plot
ax = plt.plot( k_values, coherences2 )
plt.legend(['NMF'])
plt.xticks(k_values)
plt.xlabel("Number of Topics")
plt.ylabel("Mean Coherence")
# add the points
plt.scatter( k_values, coherences2, s=120)
# find and annotate the maximum point on the plot
ymax = max(coherences2)
xpos = coherences2.argmax()
best_k = k_values[xpos]
plt.annotate( "k=%d" % best_k, xy=(best_k, ymax), xytext=(best_k, ymax), textcoords="offset points", fontsize=16)
# show the plot
#plt.show()
plt.savefig('Grafico2.jpg')


