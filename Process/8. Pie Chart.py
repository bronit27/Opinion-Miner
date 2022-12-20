#Result in the form of a piechart
mylabels=["Positive","Negative","Neutral"]
#pos =No. of Positives
#neg = No. of Negatives
#new =No. of neutrals
x=[pos,neg,new]
labels = list(mylabels)
myexplode = [0.2, 0.2, 0]
colors = ['orange',  'darkblue', 'red']# choosing colours
plt.pie(x,labels=labels,colors=colors,explode=myexplode,autopct='%1.1f%%', shadow=True, startangle=90)
plt.show()
