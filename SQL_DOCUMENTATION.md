# Documentación SQL del Proyecto ETL 🗄️

## Resumen
Este proyecto crea una base de datos SQLite con una tabla que almacena información de los bancos más grandes del mundo, incluyendo su capitalización de mercado en múltiples monedas.

---

## 📊 Base de Datos

### Nombre: `Banks.db`
- **Tipo**: SQLite Database
- **Ubicación**: `./Banks.db` (raíz del proyecto)
- **Motor**: SQLite3

---

## 📋 Tabla Creada

### Tabla: `Largest_banks`

#### Estructura de la Tabla

```sql
CREATE TABLE "Largest_banks" (
    "Name" TEXT,
    "MC_USD_Billion" REAL,
    "MC_GBP_Billion" REAL,
    "MC_EUR_Billion" REAL,
    "MC_INR_Billion" REAL
);
```

#### Descripción de Columnas

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `Name` | TEXT | Nombre del banco |
| `MC_USD_Billion` | REAL | Capitalización de mercado en USD (billones) |
| `MC_GBP_Billion` | REAL | Capitalización de mercado en GBP (billones) |
| `MC_EUR_Billion` | REAL | Capitalización de mercado en EUR (billones) |
| `MC_INR_Billion` | REAL | Capitalización de mercado en INR (billones) |

#### Características
- **Modo de inserción**: `if_exists='replace'` - La tabla se reemplaza en cada ejecución
- **Sin índice primario**: No hay clave primaria definida
- **Sin restricciones**: No hay FOREIGN KEYs ni CHECK constraints

---

## 🔍 Consultas SQL Ejecutadas

El proyecto ejecuta automáticamente **5 consultas de verificación** después de cargar los datos:

### 1. Consulta de Todos los Datos
```sql
SELECT * FROM Largest_banks
```
**Propósito**: Mostrar todos los bancos y sus capitalizaciones en todas las monedas.

### 2. Promedio de Capitalización en GBP
```sql
SELECT AVG(MC_GBP_Billion) AS Average_GBP 
FROM Largest_banks
```
**Propósito**: Calcular el promedio de capitalización de mercado en libras esterlinas.

### 3. Top 5 Nombres de Bancos
```sql
SELECT Name 
FROM Largest_banks 
LIMIT 5
```
**Propósito**: Mostrar los primeros 5 nombres de bancos en la tabla.

### 4. Conteo Total de Bancos
```sql
SELECT COUNT(*) AS Total_Banks 
FROM Largest_banks
```
**Propósito**: Contar el número total de bancos almacenados.

### 5. Top 3 Bancos por Capitalización (USD)
```sql
SELECT Name, MC_USD_Billion 
FROM Largest_banks 
ORDER BY MC_USD_Billion DESC 
LIMIT 3
```
**Propósito**: Mostrar los 3 bancos con mayor capitalización de mercado en USD.

---

## 💡 Consultas SQL Adicionales Recomendadas

Puedes ejecutar estas consultas directamente en SQLite para análisis adicionales:

### Top 10 Bancos por USD
```sql
SELECT Name, MC_USD_Billion 
FROM Largest_banks 
ORDER BY MC_USD_Billion DESC 
LIMIT 10;
```

### Comparación de Monedas para un Banco Específico
```sql
SELECT Name, 
       MC_USD_Billion AS "USD (Billions)",
       MC_GBP_Billion AS "GBP (Billions)",
       MC_EUR_Billion AS "EUR (Billions)",
       MC_INR_Billion AS "INR (Billions)"
FROM Largest_banks 
WHERE Name LIKE '%Chase%';
```

### Estadísticas por Moneda
```sql
SELECT 
    'USD' AS Currency,
    COUNT(*) AS Count,
    AVG(MC_USD_Billion) AS Average,
    MIN(MC_USD_Billion) AS Minimum,
    MAX(MC_USD_Billion) AS Maximum,
    SUM(MC_USD_Billion) AS Total
FROM Largest_banks
UNION ALL
SELECT 
    'GBP' AS Currency,
    COUNT(*) AS Count,
    AVG(MC_GBP_Billion) AS Average,
    MIN(MC_GBP_Billion) AS Minimum,
    MAX(MC_GBP_Billion) AS Maximum,
    SUM(MC_GBP_Billion) AS Total
FROM Largest_banks
UNION ALL
SELECT 
    'EUR' AS Currency,
    COUNT(*) AS Count,
    AVG(MC_EUR_Billion) AS Average,
    MIN(MC_EUR_Billion) AS Minimum,
    MAX(MC_EUR_Billion) AS Maximum,
    SUM(MC_EUR_Billion) AS Total
FROM Largest_banks;
```

### Bancos con Capitalización Mayor al Promedio
```sql
SELECT Name, MC_USD_Billion
FROM Largest_banks
WHERE MC_USD_Billion > (SELECT AVG(MC_USD_Billion) FROM Largest_banks)
ORDER BY MC_USD_Billion DESC;
```

### Ratio USD/EUR
```sql
SELECT 
    Name,
    MC_USD_Billion,
    MC_EUR_Billion,
    ROUND(MC_USD_Billion / MC_EUR_Billion, 4) AS USD_EUR_Ratio
FROM Largest_banks
ORDER BY MC_USD_Billion DESC;
```

---

## 🔧 Cómo Ejecutar Consultas SQL

### Opción 1: Desde la Terminal
```bash
sqlite3 Banks.db
```

Luego ejecuta tus consultas:
```sql
sqlite> SELECT * FROM Largest_banks;
sqlite> .quit
```

### Opción 2: Consulta Directa desde Terminal
```bash
sqlite3 Banks.db "SELECT * FROM Largest_banks;"
```

### Opción 3: Desde Python
```python
import sqlite3

conn = sqlite3.connect('Banks.db')
cursor = conn.cursor()

cursor.execute("SELECT * FROM Largest_banks")
results = cursor.fetchall()

for row in results:
    print(row)

conn.close()
```

### Opción 4: Con Pandas
```python
import pandas as pd
import sqlite3

conn = sqlite3.connect('Banks.db')
df = pd.read_sql("SELECT * FROM Largest_banks", conn)
print(df)
conn.close()
```

---

## 📈 Operaciones SQL Realizadas por el Código

### Durante la Ejecución del ETL:

1. **Conexión a la Base de Datos**
   ```python
   conn = sqlite3.connect('Banks.db')
   ```

2. **Creación/Reemplazo de Tabla**
   ```python
   df.to_sql('Largest_banks', conn, if_exists='replace', index=False)
   ```
   - Crea la tabla si no existe
   - Reemplaza completamente si ya existe

3. **Conteo de Filas (Antes)**
   ```sql
   SELECT COUNT(*) FROM Largest_banks
   ```

4. **Conteo de Filas (Después)**
   ```sql
   SELECT COUNT(*) FROM Largest_banks
   ```

5. **Commit de Transacción**
   ```python
   conn.commit()
   ```

6. **Cierre de Conexión**
   ```python
   conn.close()
   ```

---

## 🎯 Resumen de lo que se Crea

✅ **1 Base de Datos SQLite**: `Banks.db`  
✅ **1 Tabla**: `Largest_banks`  
✅ **5 Columnas**: Name, MC_USD_Billion, MC_GBP_Billion, MC_EUR_Billion, MC_INR_Billion  
✅ **5 Consultas de Verificación**: Ejecutadas automáticamente  
✅ **Context Manager**: Manejo seguro de conexiones con commit/rollback automático  

---

## 📝 Notas Importantes

- La tabla se **reemplaza completamente** en cada ejecución (`if_exists='replace'`)
- No hay **índices** creados (podrías agregar uno en `Name` si necesitas búsquedas frecuentes)
- No hay **clave primaria** (considera agregar un ID autoincremental si necesitas)
- Los datos se **validan** antes de insertarse (market cap mínimo, formato correcto, etc.)
- La base de datos usa **transacciones** para garantizar integridad

---

## 🚀 Mejoras Futuras Sugeridas

1. **Agregar Índice en Name**:
   ```sql
   CREATE INDEX idx_bank_name ON Largest_banks(Name);
   ```

2. **Agregar Clave Primaria**:
   ```sql
   ALTER TABLE Largest_banks ADD COLUMN id INTEGER PRIMARY KEY AUTOINCREMENT;
   ```

3. **Agregar Timestamp**:
   ```sql
   ALTER TABLE Largest_banks ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
   ```

4. **Crear Vista para Top 10**:
   ```sql
   CREATE VIEW top_10_banks AS
   SELECT Name, MC_USD_Billion
   FROM Largest_banks
   ORDER BY MC_USD_Billion DESC
   LIMIT 10;
   ```
