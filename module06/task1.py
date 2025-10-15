import matplotlib.pyplot as plt
sources = [ "Crude oil",  "Natural gas", "Renewable energy",
    "Solid fuels", "Nuclear energy" ]
percentages = [37.7, 20.4, 19.5, 10.6, 11.8]
fig, ax = plt.subplots()
ax.plot(sources, percentages)

ax.set_xlabel('energy source)')
ax.set_ylabel('contribution')
ax.set_title('impact of energy source')
# ax.legend()
ax.grid(True)
plt.show(block=False)
plt.savefig('energySources.png',format = 'png', dpi = 200, bbox_inches = 'tight')


fig, ax = plt.subplots()
ax.bar(sources, percentages, color = 'blue')
plt.show(block=False)
plt.close("all")
          

plt.figure(num=1, figsize=(5,5))
ax = plt.axes()
ax.bar(sources, percentages, color = 'blue')
plt.xticks(rotation=45)
plt.subplots_adjust(bottom=0.25)
plt.title('energy sources')
ax.grid(True)

plt.figure(num=2)
ax = plt.axes()
ax.plot(sources, percentages)
ax.grid(True)


colors = ['#c0392b', '#2980b9', '#27ae60', '#7f8c8d', '#f1c40f']  # palette armoniosa
explode = [0.05, 0.05, 0, 0, 0]  # leggero distacco dei primi due spicchi

plt.figure(num=3, figsize=(7,7))
plt.pie(
    percentages,
    labels=sources,
    autopct='%1.1f%%',
    startangle=90,
    counterclock=False,
    explode=explode,
    colors=colors,
    shadow=True,
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
)
plt.title("Energy Share 2023", fontsize=14, pad=20)
plt.tight_layout()
plt.show()
