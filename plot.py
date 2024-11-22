import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset('iris')
sns.set_style("white")

sns.kdeplot(x=df.sepal_width, y=df.sepal_length, cmap="Blues", fill=True, bw_adjust=0.5)
# plt.plot( 'x', 'y', "", data=df, linestyle='', marker='o')
sns.scatterplot(x=df.sepal_width, y=df.sepal_length)
plt.savefig('plot/test.png')
# plt.show()