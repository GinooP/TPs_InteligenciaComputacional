import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

tabla_pokemon = pd.read_csv('TP_Creativo/pokemon.csv')
tabla_combates_indv = pd.read_csv('TP_Creativo/combats.csv')
tabla_equipos = np.loadtxt('TP_Creativo/pokemon_id_each_team.csv',delimiter=',',skiprows=1)
tabla_team_combats = np.loadtxt('TP_Creativo/team_combat.csv',delimiter=',',skiprows=1)

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
tabla_pokemon_filtrada /= 255
#print(tabla_pokemon_filtrada)

#print(tabla_pokemon_filtrada.copy().to_numpy())

#armamos la tabla con las estadisticas filtradas y agregando la codificacion one hot
tabla_final_pokemon_stats = np.concatenate((tabla_pokemon_filtrada,datos_codificados),axis=1)
#print(tabla_final_pokemon_stats.shape[1])

tabla_combates_indv = tabla_combates_indv.to_numpy().copy()

tabla_combates_indv[:,-1] = (tabla_combates_indv[:,-1] == tabla_combates_indv[:,1]).T#uno si gana el segundo va 1, si no 0

#print(tabla_combates_indv)

#ponemos los pokemones de combates individuales
tabla_combates_indv_stats=np.concatenate((tabla_final_pokemon_stats[tabla_combates_indv[:,0]-1,:],tabla_final_pokemon_stats[tabla_combates_indv[:,1]-1,:]),axis=1)


# print(tabla_combates_indv_stats.shape)
# print(tabla_team_combats)
# print(tabla_equipos)

# tabla final pokemon -> stats
# tabla equipos -> ind pokemon -> stat1,...,stat6
# tabla combates equipos -> ind de equipo 1, ind equipo 2

tabla_team_combats = tabla_team_combats.astype(int)

tabla_ind_pokemon_eq1 = tabla_equipos[tabla_team_combats[:,0]-1,1:].astype(int)
tabla_ind_pokemon_eq2 = tabla_equipos[tabla_team_combats[:,1]-1,1:].astype(int)

# print(tabla_ind_pokemon_eq1)
# print(tabla_ind_pokemon_eq2)

tabla_stats_eq1 = tabla_final_pokemon_stats[tabla_ind_pokemon_eq1[:,0]-1,:]
tabla_stats_eq2 = tabla_final_pokemon_stats[tabla_ind_pokemon_eq2[:,0]-1,:]

for i in range(1,6):
    tabla_stats_eq1_aux = tabla_final_pokemon_stats[tabla_ind_pokemon_eq1[:,i]-1,:]
    tabla_stats_eq2_aux = tabla_final_pokemon_stats[tabla_ind_pokemon_eq2[:,i]-1,:]
    # print(tabla_stats_eq1_aux)
    # print(tabla_stats_eq1_aux.shape)
    tabla_stats_eq1 = np.concatenate((tabla_stats_eq1,tabla_stats_eq1_aux),axis=1)
    tabla_stats_eq2 = np.concatenate((tabla_stats_eq2,tabla_stats_eq2_aux),axis=1)
    # print(tabla_stats_eq1)
    # print(tabla_stats_eq1.shape)

# print(tabla_stats_eq1)
# print(tabla_stats_eq1.shape)
tabla_combates_equipos_stats = np.concatenate((tabla_stats_eq1,tabla_stats_eq2),axis=1)
# print(tabla_combates_equipos_stats)
# print(tabla_combates_equipos_stats.shape)


tabla_combates_individuales = np.concatenate((tabla_combates_indv_stats,tabla_combates_indv[:,-1:]),axis=1) # -1: es para que el arreglo sea Bidimensional
tabla_combates_grupales = np.concatenate((tabla_combates_equipos_stats,tabla_team_combats[:,-1:]),axis=1)

np.random.default_rng().shuffle(tabla_combates_grupales)

np.savetxt('TP_Creativo/combates_indv_completo.csv', tabla_combates_individuales, delimiter=',', fmt='%.10f')
np.savetxt('TP_Creativo/combates_grup_completo.csv', tabla_combates_grupales, delimiter=',', fmt='%.10f')

