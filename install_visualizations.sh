#!/bin/bash
# Script para instalar las librerías necesarias para visualizaciones

echo "Instalando librerías para visualizaciones..."
echo ""

# Activar el entorno virtual
source venv/bin/activate

# Instalar matplotlib y seaborn
pip install matplotlib seaborn

echo ""
echo "✓ Instalación completada!"
echo ""
echo "Ahora puedes ejecutar: python banks_project.py"
echo "Las visualizaciones se guardarán en la carpeta 'visualizations/'"
