import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

tabla_ids_equipos = pd.read_csv('TP_creativo/pokemon_id_each_team.csv')
tabla_combates_indv =pd.read_csv('TP_creativo/combats.csv')
tabla_pokemon = pd.read_csv('TP_creativo/pokemon.csv')
tabla_pokemon_team_combats= np.loadtxt('TP_creativo/team_combats.csv',delimiter=',',skiprows=1)

tipos = pd.concat([
    tabla_pokemon['Type 1'],
    tabla_pokemon['Type 2']
]).dropna().unique() #concatenamos todos los tipos quitando los nam y sin repetirlos (esto tiene una dimension)

encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore',categories=[tipos])
encoder.fit(tipos.reshape(-1,1)) #hacemos que codifica con las etiquetas type 1 y type 2 codificadas

#tipos [fuego,agua]
#tipos [[fuego],[agua]]
#print(tabla_pokemon)

datos_codificados_1 = encoder.transform(tabla_pokemon[['Type 1']])#codificamos el primer tipo

datos_codificados_2= encoder.transform(tabla_pokemon[['Type 2']].fillna('Sin tipo'))#codificamos el tipo 2 con los datos nam como 'sin tipo' para ponga todo en cero en esos casos

datos_codificados = datos_codificados_1 + datos_codificados_2

tabla_pokemon_filtrada = tabla_pokemon.drop(columns=['Generation','Legendary','#','Name','Type 1','Type 2'])
#tabla_pokemon_filtrada /= 255
#print(tabla_pokemon_filtrada)

#print(tabla_pokemon_filtrada.copy().to_numpy())

#armamos la tabla con las estadisticas filtradas y agregando la codificacion one hot
tabla_final_pokemon_stats=np.concatenate((tabla_pokemon_filtrada,datos_codificados),axis=1)
#print(tabla_final_pokemon_stats.shape[1])

tabla_combates_indv = tabla_combates_indv.to_numpy().copy()

tabla_combates_indv[:,-1] = (tabla_combates_indv[:,-1] == tabla_combates_indv[:,1]).T#uno si gana el segundo va 1, si no 0

#print(tabla_combates_indv)

#ponemos los pokemones de combates individuales
tabla_combates_indv_stats=np.concatenate((tabla_final_pokemon_stats[tabla_combates_indv[:,0]-1,:],tabla_final_pokemon_stats[tabla_combates_indv[:,1]-1,:]),axis=1)


#print(tabla_combates_indv_stats.shape)
print(tabla_pokemon_team_combats)
