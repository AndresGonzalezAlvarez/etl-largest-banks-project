# 📊 Instalación de Visualizaciones

## Problema: Las visualizaciones no aparecen

Si las visualizaciones no se están generando, es porque las librerías necesarias no están instaladas.

## ✅ Solución Rápida

### Opción 1: Usar el script de instalación (Recomendado)

```bash
./install_visualizations.sh
```

### Opción 2: Instalación manual

1. Activa el entorno virtual:
```bash
source venv/bin/activate
```

2. Instala las librerías:
```bash
pip install matplotlib seaborn
```

3. Ejecuta el proyecto:
```bash
python banks_project.py
```

### Opción 3: Instalación directa con pip

```bash
venv/bin/pip install matplotlib seaborn
```

## 📁 Dónde encontrar las visualizaciones

Una vez instaladas las librerías y ejecutado el proyecto, las visualizaciones se guardarán en:

```
./visualizations/
```

### Archivos generados:

1. `01_top_banks_usd.png` - Top 10 bancos por capitalización USD
2. `02_currency_comparison.png` - Comparación de monedas
3. `03_market_cap_distribution.png` - Distribución de capitalización
4. `04_currency_heatmap.png` - Mapa de calor de monedas
5. `05_comprehensive_dashboard.png` - Dashboard completo

## 🔍 Verificar instalación

Para verificar que las librerías están instaladas:

```bash
python3 -c "import matplotlib; import seaborn; print('✓ Librerías instaladas correctamente')"
```

Si ves el mensaje de éxito, las librerías están instaladas.

## ⚠️ Solución de problemas

### Error: "ModuleNotFoundError: No module named 'matplotlib'"

**Solución**: Instala las librerías usando uno de los métodos arriba.

### Error: "Permission denied" al ejecutar el script

**Solución**: 
```bash
chmod +x install_visualizations.sh
./install_visualizations.sh
```

### Las visualizaciones no se generan pero no hay error

**Verifica**:
1. ¿Están instaladas las librerías? (ver sección "Verificar instalación")
2. ¿Se ejecutó el proyecto completamente? Revisa `code_log.txt`
3. ¿Existe la carpeta `visualizations/`? Si no existe, se creará automáticamente.

### El proyecto se detiene antes de las visualizaciones

**Causa**: Hay un error en una fase anterior (extracción, transformación o carga).

**Solución**: 
- Revisa el archivo `code_log.txt` para ver qué fase falló
- El código ahora continúa con las visualizaciones incluso si hay problemas menores en la base de datos

## 📝 Notas

- Las visualizaciones se generan **después** de todas las fases del ETL
- Si hay un error crítico en extracción o transformación, el proceso se detiene antes de las visualizaciones
- Las visualizaciones requieren que los datos estén transformados correctamente
