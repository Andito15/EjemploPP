#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <float.h>

#define INITIAL_CAPACITY 1000
#define UMBRAL 20.0

int main() {
    FILE *archivo = fopen("datos.txt", "r");
    if (archivo == NULL) {
        printf("Error: no se pudo abrir el archivo datos.txt\n");
        return 1;
    }

    int capacidad = INITIAL_CAPACITY;
    int n = 0;
    double *datos = (double *)malloc(capacidad * sizeof(double));
    if (datos == NULL) {
        printf("Error: no se pudo reservar memoria\n");
        fclose(archivo);
        return 1;
    }

    while (fscanf(archivo, "%lf", &datos[n]) == 1) {
        n++;
        if (n >= capacidad) {
            capacidad *= 2;
            double *temp = (double *)realloc(datos, capacidad * sizeof(double));
            if (temp == NULL) {
                printf("Error: no se pudo ampliar memoria\n");
                free(datos);
                fclose(archivo);
                return 1;
            }
            datos = temp;
        }
    }
    fclose(archivo);

    if (n == 0) {
        printf("El archivo no contiene datos válidos.\n");
        free(datos);
        return 1;
    }

    double suma = 0.0;
    double minimo = DBL_MAX;
    double maximo = -DBL_MAX;
    int mayores_umbral = 0;

    double inicio = omp_get_wtime();

    #pragma omp parallel
    {
        double min_local = DBL_MAX;
        double max_local = -DBL_MAX;

        #pragma omp for reduction(+:suma, mayores_umbral)
        for (int i = 0; i < n; i++) {
            suma += datos[i];

            if (datos[i] > UMBRAL) {
                mayores_umbral++;
            }

            if (datos[i] < min_local) {
                min_local = datos[i];
            }

            if (datos[i] > max_local) {
                max_local = datos[i];
            }
        }

        #pragma omp critical
        {
            if (min_local < minimo) {
                minimo = min_local;
            }
            if (max_local > maximo) {
                maximo = max_local;
            }
        }
    }

    double fin = omp_get_wtime();

    double promedio = suma / n;

    printf("Cantidad de datos: %d\n", n);
    printf("Suma total: %.2f\n", suma);
    printf("Promedio: %.2f\n", promedio);
    printf("Valor mínimo: %.2f\n", minimo);
    printf("Valor máximo: %.2f\n", maximo);
    printf("Mayores que %.2f: %d\n", UMBRAL, mayores_umbral);
    printf("Tiempo paralelo: %.6f segundos\n", fin - inicio);
    printf("Hilos utilizados: %d\n", omp_get_max_threads());

    free(datos);
    return 0;
}
