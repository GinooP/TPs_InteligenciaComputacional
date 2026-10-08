import numpy as np

import enjambre_particulas as enj

#f1=lambda x: -x * np.sin(abs(x)**0.5)

def f1(particula):
    x=particula[0]

    y= -x * np.sin(abs(x)**0.5)
    return y


def f2(particula):
    x1=particula[0]
    x2=particula[1]

    y = (((x1**2) + (x2**2))**0.25) * ((np.sin(50*(((x1**2) + (x2**2))**0.1))**2) + 1)
    
    return y



cant_particulas1=30
it_max1=100

entorno1= 2

condiciones_iniciales_1=np.array([[-512,512]])

tol1= -415
aceleracion_1=[2,2]

cant_particulas2=30
it_max2=1000

entorno2= 2

condiciones_iniciales_2=np.array([[-100,100],[-100,100]])
tol2= 1e-3
aceleracion_2=[1,1]

# mejor_particula = enj.enjambre_particulas(cant_particulas=cant_particulas1,f=f1,entorno=entorno1,condiciones_iniciales=condiciones_iniciales_1,
#                                           it_max=it_max1,tol=tol1,aceleracion=aceleracion_1)

# #print(f"esta es la mejor particula:{mejor_particula} || valor obtenido: {f1(mejor_particula)}")

mejor_particula = enj.enjambre_particulas(cant_particulas=cant_particulas2,f=f2,entorno=entorno2,condiciones_iniciales=condiciones_iniciales_2,
                                          it_max=it_max2,tol=tol2,aceleracion=aceleracion_2)

print(f"esta es la mejor particula:{mejor_particula}")

print(f"valor obtenido: {f2(mejor_particula)}")