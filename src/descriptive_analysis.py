"""
Módulo de análisis descriptivo para el mercado inmobiliario danés.
Contiene todas las funciones necesarias para realizar análisis completo de KPIs regionales,
precios por m², volumen de transacciones y tendencias temporales.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h2o
from datetime import datetime
from scipy import stats
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# FUNCIONES DE UTILIDAD Y CONFIGURACIÓN
# =============================================================================

def load_and_validate_data(data_path, destination_frame='df_clean'):
    """
    Cargar y validar datos del análisis exploratorio usando H2O.
    
    Parameters:
    -----------
    data_path : str
        Ruta al archivo de datos limpio
    destination_frame : str
        Nombre del frame de destino en H2O
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con los datos cargados y validados
    """
    try:
        h2o.init()
        h2o.connect()
        print(f"Importando datos desde {data_path}\n")

        df_h2o = h2o.import_file(
            path=str(data_path),
            header=1,
            sep=",",
            destination_frame=str(destination_frame)
        )

        df_clean = df_h2o.as_data_frame()
        print(f"Datos importados a H2O con destino: {destination_frame}\n")
        print(f"Dimensiones del H2OFrame: {df_h2o.nrows:,} filas × {df_h2o.ncols} columnas\n")
        print(f"Datos cargados: {df_clean.shape[0]:,} registros x {df_clean.shape[1]} columnas")
        print(f"Período: {df_clean['date'].min()} - {df_clean['date'].max()}")
        print(f"Regiones: {df_clean['region'].nunique()}")
        print(f"Rango precios: {df_clean['purchase_price'].min():,.0f} - {df_clean['purchase_price'].max():,.0f} DKK")
        return df_clean

    except Exception as e:
        print(f"Error al cargar datos: {e}")
        raise


def calculate_confidence_interval(data, confidence=0.95):
    """
    Calcular intervalo de confianza para una serie de datos.
    
    Parameters:
    -----------
    data : pd.Series
        Serie de datos numéricos
    confidence : float
        Nivel de confianza (por defecto 0.95 para 95%)
        
    Returns:
    --------
    tuple
        Tupla con (límite_inferior, límite_superior)
    """
    mean = data.mean()
    sem = stats.sem(data)
    h = sem * stats.t.ppf((1 + confidence) / 2., len(data)-1)
    return mean - h, mean + h


def classify_market_size(participation):
    """
    Clasificar regiones por tamaño de mercado basado en participación.
    
    Parameters:
    -----------
    participation : float
        Porcentaje de participación en el mercado
        
    Returns:
    --------
    str
        Clasificación del mercado
    """
    if participation >= 5.0:
        return 'Principal'
    elif participation >= 2.0:
        return 'Secundario'
    elif participation >= 0.5:
        return 'Terciario'
    else:
        return 'Nicho'


def configure_plot_style():
    """Configurar estilo de visualizaciones."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")


# =============================================================================
# ANÁLISIS REGIONAL DE PRECIOS
# =============================================================================

def analyze_regional_prices(df):
    """
    Análisis completo de precios por región.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos de transacciones inmobiliarias
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con estadísticas regionales completas
    """
    # Estadísticas descriptivas por región
    regional_stats = df.groupby('region')['purchase_price'].agg([
        'count', 'mean', 'median', 'std',
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75),
        'min', 'max'
    ]).round(0)
    
    regional_stats.columns = ['Transacciones', 'Promedio', 'Mediana', 'Std', 'Q1', 'Q3', 'Minimo', 'Maximo']
    
    # Calcular intervalos de confianza
    ci_data = []
    for region in df['region'].unique():
        region_prices = df[df['region'] == region]['purchase_price']
        ci_lower, ci_upper = calculate_confidence_interval(region_prices)
        ci_data.append((ci_lower, ci_upper))
    
    ci_df = pd.DataFrame(ci_data, index=regional_stats.index, 
                        columns=['CI_Lower', 'CI_Upper']).round(0)
    
    # Combinar estadísticas
    regional_stats = pd.concat([regional_stats, ci_df], axis=1)
    regional_stats = regional_stats.sort_values('Promedio', ascending=False)
    
    return regional_stats


def print_regional_summary(regional_stats, top_n=10):
    """
    Imprimir resumen de estadísticas regionales.
    
    Parameters:
    -----------
    regional_stats : pd.DataFrame
        DataFrame con estadísticas regionales
    top_n : int
        Número de regiones top a mostrar
        
    Returns:
    --------
    pd.DataFrame
        Top N regiones
    """
    print("ESTADÍSTICAS DE PRECIOS POR REGIÓN")
    print("=" * 50)
    print(f"Total regiones analizadas: {len(regional_stats)}")
    print(f"Rango precios promedio: {regional_stats['Promedio'].min():,.0f} - {regional_stats['Promedio'].max():,.0f} DKK")
    print(f"\nTOP {top_n} REGIONES MÁS CARAS")
    print("-" * 40)
    return regional_stats.head(top_n)


def create_regional_price_plots(regional_stats, df, figsize=(16, 12)):
    """
    Crear visualizaciones de precios regionales.
    
    Parameters:
    -----------
    regional_stats : pd.DataFrame
        DataFrame con estadísticas regionales
    df : pd.DataFrame
        DataFrame original con todos los datos
    figsize : tuple
        Tamaño de la figura
        
    Returns:
    --------
    pd.Series
        Serie con coeficientes de variación
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Análisis de Precios por Región en Dinamarca', fontsize=16, fontweight='bold')

    # 1. Ranking de precios promedio (Top 15)
    top_15 = regional_stats.head(15)
    bars1 = axes[0,0].barh(range(len(top_15)), top_15['Promedio'], 
                           color=sns.color_palette("viridis", len(top_15)))
    axes[0,0].set_yticks(range(len(top_15)))
    axes[0,0].set_yticklabels(top_15.index, fontsize=10)
    axes[0,0].set_xlabel('Precio Promedio (DKK)')
    axes[0,0].set_title('Top 15 Regiones por Precio Promedio')
    axes[0,0].grid(axis='x', alpha=0.3)

    # Valores en barras
    for i, v in enumerate(top_15['Promedio']):
        axes[0,0].text(v + 50000, i, f'{v:,.0f}', va='center', fontsize=9)

    # 2. Promedio vs Mediana (Top 10)
    top_10 = regional_stats.head(10)
    x = np.arange(len(top_10))
    width = 0.35

    axes[0,1].bar(x - width/2, top_10['Promedio'], width, label='Promedio', alpha=0.8)
    axes[0,1].bar(x + width/2, top_10['Mediana'], width, label='Mediana', alpha=0.8)
    axes[0,1].set_xlabel('Regiones')
    axes[0,1].set_ylabel('Precio (DKK)')
    axes[0,1].set_title('Promedio vs Mediana - Top 10')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(top_10.index, rotation=45, ha='right', fontsize=9)
    axes[0,1].legend()
    axes[0,1].grid(axis='y', alpha=0.3)

    # 3. Distribución de precios promedio
    axes[1,0].hist(regional_stats['Promedio'], bins=20, alpha=0.7, edgecolor='black')
    mean_price = regional_stats['Promedio'].mean()
    median_price = regional_stats['Promedio'].median()
    
    axes[1,0].axvline(mean_price, color='red', linestyle='--', 
                      label=f'Media: {mean_price:,.0f} DKK')
    axes[1,0].axvline(median_price, color='orange', linestyle='--',
                      label=f'Mediana: {median_price:,.0f} DKK')
    axes[1,0].set_xlabel('Precio Promedio (DKK)')
    axes[1,0].set_ylabel('Número de Regiones')
    axes[1,0].set_title('Distribución de Precios Promedio')
    axes[1,0].legend()
    axes[1,0].grid(alpha=0.3)

    # 4. Coeficiente de variación
    cv_data = ((regional_stats['Std'] / regional_stats['Promedio']) * 100).sort_values(ascending=False).head(15)
    axes[1,1].bar(range(len(cv_data)), cv_data, alpha=0.8)
    axes[1,1].set_xticks(range(len(cv_data)))
    axes[1,1].set_xticklabels(cv_data.index, rotation=45, ha='right', fontsize=9)
    axes[1,1].set_ylabel('Coeficiente de Variación (%)')
    axes[1,1].set_title('Variabilidad por Región (Top 15)')
    axes[1,1].grid(axis='y', alpha=0.3)

    # Valores en barras
    for i, v in enumerate(cv_data):
        axes[1,1].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()
    
    return cv_data


def print_regional_insights(regional_stats, cv_data):
    """
    Imprimir insights clave del análisis regional.
    
    Parameters:
    -----------
    regional_stats : pd.DataFrame
        DataFrame con estadísticas regionales
    cv_data : pd.Series
        Serie con coeficientes de variación
    """
    print("\nINSIGHTS CLAVE - PRECIOS REGIONALES")
    print("=" * 40)
    print(f"Región más cara: {regional_stats.index[0]} ({regional_stats.iloc[0]['Promedio']:,.0f} DKK)")
    print(f"Región más económica: {regional_stats.index[-1]} ({regional_stats.iloc[-1]['Promedio']:,.0f} DKK)")
    print(f"Ratio precio max/min: {regional_stats.iloc[0]['Promedio'] / regional_stats.iloc[-1]['Promedio']:.1f}x")
    print(f"Regiones sobre la media: {(regional_stats['Promedio'] > regional_stats['Promedio'].mean()).sum()}")
    print(f"CV promedio: {cv_data.mean():.1f}%")


# =============================================================================
# ANÁLISIS DE PRECIO POR M²
# =============================================================================

def analyze_sqm_prices(df):
    """
    Análisis de precio por m² por región.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos de transacciones
        
    Returns:
    --------
    tuple
        (sqm_stats, premium_threshold)
    """
    # Estadísticas básicas
    sqm_stats = df.groupby('region')['sqm_price'].agg([
        'count', 'mean', 'median', 'std',
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75),
        'min', 'max'
    ]).round(0)
    
    sqm_stats.columns = ['Transacciones', 'Promedio_m2', 'Mediana_m2', 'Std_m2', 
                        'Q1_m2', 'Q3_m2', 'Min_m2', 'Max_m2']
    sqm_stats = sqm_stats.sort_values('Promedio_m2', ascending=False)
    
    # Identificar mercados premium
    premium_threshold = df['sqm_price'].quantile(0.75)
    sqm_stats['Es_Premium'] = sqm_stats['Promedio_m2'] > premium_threshold
    
    # Coeficiente de variación
    sqm_stats['CV_m2'] = (sqm_stats['Std_m2'] / sqm_stats['Promedio_m2']) * 100
    
    return sqm_stats, premium_threshold


def create_ranking_comparison(regional_stats, sqm_stats):
    """
    Crear comparación de rankings entre precio total y precio/m².
    
    Parameters:
    -----------
    regional_stats : pd.DataFrame
        Estadísticas regionales de precio total
    sqm_stats : pd.DataFrame
        Estadísticas regionales de precio por m²
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con comparación de rankings
    """
    comparison = pd.DataFrame({
        'Region': regional_stats.index,
        'Precio_Total': regional_stats['Promedio'].values,
        'Precio_m2': sqm_stats.loc[regional_stats.index, 'Promedio_m2'].values,
        'Rank_Total': range(1, len(regional_stats) + 1),
        'Rank_m2': range(1, len(sqm_stats) + 1)
    })
    comparison['Diferencia_Rank'] = comparison['Rank_Total'] - comparison['Rank_m2']
    return comparison


def create_sqm_price_plots(df, sqm_stats, premium_threshold, comparison_df, figsize=(16, 12)):
    """
    Crear visualizaciones de análisis de precio por m².
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame original
    sqm_stats : pd.DataFrame
        Estadísticas de precio por m²
    premium_threshold : float
        Umbral para mercados premium
    comparison_df : pd.DataFrame
        DataFrame con comparación de rankings
    figsize : tuple
        Tamaño de la figura
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Análisis de Precio por m² por Región', fontsize=16, fontweight='bold')

    # 1. Top 15 regiones por precio/m²
    top_15_sqm = sqm_stats.head(15)
    colors = ['red' if premium else 'blue' for premium in top_15_sqm['Es_Premium']]
    
    bars1 = ax1.bar(range(len(top_15_sqm)), top_15_sqm['Promedio_m2'], color=colors, alpha=0.7)
    ax1.set_title('Top 15 Regiones - Precio por m²')
    ax1.set_ylabel('Precio por m² (DKK)')
    ax1.set_xlabel('Región')
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.set_xticks(range(len(top_15_sqm)))
    ax1.set_xticklabels(top_15_sqm.index, ha='right')
    ax1.axhline(y=premium_threshold, color='red', linestyle='--', alpha=0.7, 
               label=f'Umbral Premium ({premium_threshold:,.0f})')
    ax1.legend()

    # 2. Comparación normalizada
    top_10_comp = comparison_df.head(10)
    x_pos = np.arange(len(top_10_comp))
    width = 0.35
    
    # Normalizar para comparación visual
    precio_total_norm = (top_10_comp['Precio_Total'] / top_10_comp['Precio_Total'].max()) * 100
    precio_m2_norm = (top_10_comp['Precio_m2'] / top_10_comp['Precio_m2'].max()) * 100
    
    ax2.bar(x_pos - width/2, precio_total_norm, width, label='Precio Total', alpha=0.8)
    ax2.bar(x_pos + width/2, precio_m2_norm, width, label='Precio/m²', alpha=0.8)
    ax2.set_title('Comparación Normalizada: Precio Total vs Precio/m²')
    ax2.set_ylabel('Valor Normalizado (%)')
    ax2.set_xlabel('Región (Top 10)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(top_10_comp['Region'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Distribución nacional de precio/m²
    ax3.hist(df['sqm_price'], bins=50, alpha=0.7, edgecolor='black')
    ax3.axvline(df['sqm_price'].mean(), color='red', linestyle='-', linewidth=2, 
               label=f'Media: {df["sqm_price"].mean():,.0f}')
    ax3.axvline(df['sqm_price'].median(), color='green', linestyle='--', linewidth=2,
               label=f'Mediana: {df["sqm_price"].median():,.0f}')
    ax3.axvline(premium_threshold, color='orange', linestyle=':', linewidth=2,
               label=f'Umbral Premium: {premium_threshold:,.0f}')
    ax3.set_title('Distribución Nacional de Precio por m²')
    ax3.set_xlabel('Precio por m² (DKK)')
    ax3.set_ylabel('Frecuencia')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Variabilidad por región
    top_cv = sqm_stats.sort_values('CV_m2', ascending=False).head(15)
    ax4.bar(range(len(top_cv)), top_cv['CV_m2'], alpha=0.8)
    ax4.set_title('Variabilidad del Precio/m² por Región')
    ax4.set_ylabel('Coeficiente de Variación (%)')
    ax4.set_xlabel('Región')
    ax4.set_xticks(range(len(top_cv)))
    ax4.set_xticklabels(top_cv.index, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)

    # Valores en barras
    for i, v in enumerate(top_cv['CV_m2']):
        ax4.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()


def print_sqm_insights(sqm_stats, comparison_df):
    """
    Imprimir insights del análisis de precio por m².
    
    Parameters:
    -----------
    sqm_stats : pd.DataFrame
        Estadísticas de precio por m²
    comparison_df : pd.DataFrame
        Comparación de rankings
    """
    print("\nINSIGHTS CLAVE - PRECIO POR M²")
    print("=" * 35)
    print(f"Región más eficiente: {sqm_stats.index[0]}")
    print(f"Precio/m² máximo: {sqm_stats.iloc[0]['Promedio_m2']:,.0f} DKK/m²")
    top_cv = sqm_stats.sort_values('CV_m2', ascending=False)
    print(f"Región más variable: {top_cv.index[0]} ({top_cv.iloc[0]['CV_m2']:.1f}%)")
    print(f"Región más estable: {sqm_stats.sort_values('CV_m2').index[0]} ({sqm_stats.sort_values('CV_m2').iloc[0]['CV_m2']:.1f}%)")
    print(f"Mayor diferencia ranking: {abs(comparison_df['Diferencia_Rank']).max()} posiciones")


# =============================================================================
# ANÁLISIS DE VOLUMEN DE TRANSACCIONES
# =============================================================================

def analyze_transaction_volume(df):
    """
    Análisis de volumen de transacciones por región.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos de transacciones
        
    Returns:
    --------
    tuple
        (volume_stats, correlation, high_liquidity_threshold)
    """
    # Estadísticas de volumen
    volume_stats = df.groupby('region').agg({
        'purchase_price': ['count', 'sum'],
        'sqm_price': 'mean'
    }).round(0)
    
    volume_stats.columns = ['Num_Transacciones', 'Volumen_Total_DKK', 'Precio_Promedio_m2']
    volume_stats = volume_stats.sort_values('Num_Transacciones', ascending=False)
    
    # Participación de mercado
    total_transactions = volume_stats['Num_Transacciones'].sum()
    volume_stats['Participacion_Mercado'] = (volume_stats['Num_Transacciones'] / total_transactions) * 100
    volume_stats['Participacion_Acumulada'] = volume_stats['Participacion_Mercado'].cumsum()
    
    # Clasificación por tamaño de mercado
    volume_stats['Tipo_Mercado'] = volume_stats['Participacion_Mercado'].apply(classify_market_size)
    
    # Correlación volumen-precio
    correlation = np.corrcoef(volume_stats['Num_Transacciones'], volume_stats['Precio_Promedio_m2'])[0,1]
    
    # Alta liquidez (top 20%)
    high_liquidity_threshold = volume_stats['Num_Transacciones'].quantile(0.8)
    volume_stats['Alta_Liquidez'] = volume_stats['Num_Transacciones'] > high_liquidity_threshold
    
    return volume_stats, correlation, high_liquidity_threshold


def print_volume_summary(volume_stats, correlation, high_liquidity_threshold):
    """
    Imprimir resumen del análisis de volumen.
    
    Parameters:
    -----------
    volume_stats : pd.DataFrame
        Estadísticas de volumen
    correlation : float
        Correlación volumen-precio
    high_liquidity_threshold : float
        Umbral de alta liquidez
    """
    print("ANÁLISIS DE VOLUMEN DE TRANSACCIONES")
    print("=" * 45)
    print(f"Total transacciones: {volume_stats['Num_Transacciones'].sum():,}")
    print(f"Volumen total: {volume_stats['Volumen_Total_DKK'].sum():,.0f} DKK")
    print(f"Correlación volumen-precio/m²: {correlation:.3f}")
    print(f"Regiones alta liquidez: {volume_stats['Alta_Liquidez'].sum()}")
    
    # Análisis de concentración
    pareto_80 = volume_stats[volume_stats['Participacion_Acumulada'] <= 80]
    print(f"\nCONCENTRACIÓN DE MERCADO")
    print("-" * 25)
    print(f"Regiones que concentran 80% del mercado: {len(pareto_80)}")
    print(f"Participación top 10: {volume_stats.head(10)['Participacion_Mercado'].sum():.1f}%")
    
    # Distribución por tipo
    market_dist = volume_stats['Tipo_Mercado'].value_counts()
    print(f"\nDISTRIBUCIÓN POR TIPO DE MERCADO")
    print("-" * 30)
    for market_type, count in market_dist.items():
        percentage = (count / len(volume_stats)) * 100
        print(f"{market_type}: {count} regiones ({percentage:.1f}%)")


def create_volume_plots(volume_stats, correlation, high_liquidity_threshold, figsize=(16, 12)):
    """
    Crear visualizaciones de análisis de volumen.
    
    Parameters:
    -----------
    volume_stats : pd.DataFrame
        Estadísticas de volumen
    correlation : float
        Correlación volumen-precio
    high_liquidity_threshold : float
        Umbral de alta liquidez
    figsize : tuple
        Tamaño de la figura
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Análisis de Volumen de Transacciones por Región', fontsize=16, fontweight='bold')

    # Mapeo de colores para tipos de mercado
    market_colors = {'Principal': 'red', 'Secundario': 'blue', 'Terciario': 'green', 'Nicho': 'gray'}

    # 1. Top 20 regiones por volumen
    top_20 = volume_stats.head(20)
    colors = [market_colors[market] for market in top_20['Tipo_Mercado']]
    
    bars1 = ax1.bar(range(len(top_20)), top_20['Num_Transacciones'], color=colors, alpha=0.7)
    ax1.set_title('Top 20 Regiones por Volumen de Transacciones')
    ax1.set_ylabel('Número de Transacciones')
    ax1.set_xlabel('Región')
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.set_xticks(range(len(top_20)))
    ax1.set_xticklabels(top_20.index, ha='right')
    ax1.axhline(y=high_liquidity_threshold, color='red', linestyle='--', alpha=0.7, 
               label=f'Umbral Alta Liquidez ({high_liquidity_threshold:,.0f})')
    ax1.legend()

    # 2. Diagrama de Pareto
    x_pos = range(len(top_20))
    ax2_twin = ax2.twinx()
    
    # Barras de participación individual
    ax2.bar(x_pos, top_20['Participacion_Mercado'], alpha=0.7, label='Participación Individual')
    # Línea de participación acumulada
    ax2_twin.plot(x_pos, top_20['Participacion_Acumulada'], color='red', marker='o', 
                  linewidth=2, label='Participación Acumulada')
    
    ax2.set_title('Diagrama de Pareto - Concentración del Mercado')
    ax2.set_ylabel('Participación Individual (%)', color='blue')
    ax2_twin.set_ylabel('Participación Acumulada (%)', color='red')
    ax2.set_xlabel('Región (Top 20)')
    ax2.set_xticks(x_pos[::2])
    ax2.set_xticklabels(top_20.index[::2], rotation=45, ha='right')
    ax2_twin.axhline(y=80, color='green', linestyle=':', alpha=0.7, label='Regla 80/20')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')

    # 3. Relación Volumen vs Precio/m²
    scatter_colors = [market_colors[market] for market in volume_stats['Tipo_Mercado']]
    ax3.scatter(volume_stats['Num_Transacciones'], volume_stats['Precio_Promedio_m2'], 
               c=scatter_colors, alpha=0.7, s=50)
    ax3.set_xscale('log')
    ax3.set_title(f'Relación Volumen vs Precio/m² (r={correlation:.3f})')
    ax3.set_xlabel('Número de Transacciones (escala log)')
    ax3.set_ylabel('Precio Promedio por m² (DKK)')
    ax3.grid(True, alpha=0.3)
    
    # Línea de tendencia
    log_volume = np.log10(volume_stats['Num_Transacciones'])
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_volume, volume_stats['Precio_Promedio_m2'])
    line_x = np.linspace(volume_stats['Num_Transacciones'].min(), volume_stats['Num_Transacciones'].max(), 100)
    line_y = slope * np.log10(line_x) + intercept
    ax3.plot(line_x, line_y, 'r--', alpha=0.8, label=f'Tendencia (R²={r_value**2:.3f})')
    ax3.legend()

    # 4. Distribución de tipos de mercado
    market_counts = volume_stats['Tipo_Mercado'].value_counts()
    colors_pie = [market_colors[market] for market in market_counts.index]
    
    wedges, texts, autotexts = ax4.pie(market_counts.values, labels=market_counts.index, 
                                      colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Distribución de Regiones por Tipo de Mercado')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)

    plt.tight_layout()
    plt.show()


def print_volume_insights(volume_stats, correlation):
    """
    Imprimir insights del análisis de volumen.
    
    Parameters:
    -----------
    volume_stats : pd.DataFrame
        Estadísticas de volumen
    correlation : float
        Correlación volumen-precio
    """
    print("\nINSIGHTS CLAVE - VOLUMEN DE TRANSACCIONES")
    print("=" * 45)
    print(f"Región líder: {volume_stats.index[0]} ({volume_stats.iloc[0]['Num_Transacciones']:,} trans.)")
    print(f"Participación del líder: {volume_stats.iloc[0]['Participacion_Mercado']:.1f}%")
    print(f"Mercados principales: {(volume_stats['Tipo_Mercado'] == 'Principal').sum()} regiones")
    print(f"Concentración top 5: {volume_stats.head(5)['Participacion_Mercado'].sum():.1f}%")
    print(f"Correlación volumen-precio: {'Positiva' if correlation > 0 else 'Negativa'} ({abs(correlation):.3f})")


# =============================================================================
# ANÁLISIS TEMPORAL
# =============================================================================

def analyze_temporal_trends(df):
    """
    Análisis de tendencias temporales de precios.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos de transacciones
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con estadísticas anuales
    """
    # Crear columnas de fecha
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['month'] = pd.to_datetime(df['date']).dt.month
    
    # Estadísticas anuales
    yearly_stats = df.groupby('year').agg({
        'purchase_price': ['count', 'mean', 'median', 'std'],
        'sqm_price': ['mean', 'median']
    }).round(0)
    
    yearly_stats.columns = ['Transacciones', 'Precio_Promedio', 'Precio_Mediana', 'Precio_Std',
                           'Precio_m2_Promedio', 'Precio_m2_Mediana']
    
    # Calcular tasas de crecimiento anual
    yearly_stats['Crecimiento_Precio'] = yearly_stats['Precio_Promedio'].pct_change() * 100
    yearly_stats['Crecimiento_m2'] = yearly_stats['Precio_m2_Promedio'].pct_change() * 100
    
    return yearly_stats


def create_temporal_plots(yearly_stats, figsize=(16, 10)):
    """
    Crear visualizaciones de tendencias temporales.
    
    Parameters:
    -----------
    yearly_stats : pd.DataFrame
        Estadísticas anuales
    figsize : tuple
        Tamaño de la figura
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Evolución Temporal del Mercado Inmobiliario Danés', fontsize=16, fontweight='bold')
    
    years = yearly_stats.index
    
    # 1. Evolución de precios promedio
    ax1.plot(years, yearly_stats['Precio_Promedio'], marker='o', linewidth=2, markersize=4)
    ax1.set_title('Evolución del Precio Promedio')
    ax1.set_xlabel('Año')
    ax1.set_ylabel('Precio Promedio (DKK)')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Evolución de precio por m²
    ax2.plot(years, yearly_stats['Precio_m2_Promedio'], marker='s', linewidth=2, 
             markersize=4, color='orange')
    ax2.set_title('Evolución del Precio por m²')
    ax2.set_xlabel('Año')
    ax2.set_ylabel('Precio por m² (DKK)')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Volumen de transacciones
    ax3.bar(years, yearly_stats['Transacciones'], alpha=0.7, color='green')
    ax3.set_title('Volumen de Transacciones por Año')
    ax3.set_xlabel('Año')
    ax3.set_ylabel('Número de Transacciones')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Tasas de crecimiento
    ax4.plot(years[1:], yearly_stats['Crecimiento_Precio'].dropna(), marker='o', 
             linewidth=2, label='Precio Total', alpha=0.8)
    ax4.plot(years[1:], yearly_stats['Crecimiento_m2'].dropna(), marker='s', 
             linewidth=2, label='Precio/m²', alpha=0.8)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax4.set_title('Tasas de Crecimiento Anual')
    ax4.set_xlabel('Año')
    ax4.set_ylabel('Crecimiento (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def print_temporal_insights(yearly_stats):
    """
    Imprimir insights del análisis temporal.
    
    Parameters:
    -----------
    yearly_stats : pd.DataFrame
        Estadísticas anuales
    """
    print("INSIGHTS TEMPORALES")
    print("=" * 25)
    
    # Períodos de mayor crecimiento
    max_growth_year = yearly_stats['Crecimiento_Precio'].idxmax()
    max_growth_rate = yearly_stats['Crecimiento_Precio'].max()
    
    # Períodos de mayor declive
    min_growth_year = yearly_stats['Crecimiento_Precio'].idxmin()
    min_growth_rate = yearly_stats['Crecimiento_Precio'].min()
    
    print(f"Mayor crecimiento: {max_growth_year} ({max_growth_rate:.1f}%)")
    print(f"Mayor declive: {min_growth_year} ({min_growth_rate:.1f}%)")
    print(f"Crecimiento promedio anual: {yearly_stats['Crecimiento_Precio'].mean():.1f}%")
    print(f"Período analizado: {yearly_stats.index.min()} - {yearly_stats.index.max()}")


# =============================================================================
# FUNCIÓN PRINCIPAL DE ANÁLISIS COMPLETO
# =============================================================================

def run_complete_descriptive_analysis(df, target='purchase_price', include_visualizations=True):
    """
    Función principal para ejecutar análisis descriptivo completo.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos de transacciones inmobiliarias
    target : str
        Columna objetivo (precio)
    include_visualizations : bool
        Si incluir visualizaciones
        
    Returns:
    --------
    dict
        Diccionario con todos los resultados del análisis
    """
    print("=" * 60)
    print("ANÁLISIS DESCRIPTIVO COMPLETO - MERCADO INMOBILIARIO DANÉS")
    print("=" * 60)
    
    # Configurar estilo de plots
    configure_plot_style()
    
    all_results = {}
    
    # 1. Análisis Regional
    print("\n1. ANÁLISIS REGIONAL")
    print("=" * 20)
    
    regional_stats = analyze_regional_prices(df, target)
    top_regions = print_regional_summary(regional_stats)
    cv_data = create_regional_price_plots(regional_stats, df)
    print_regional_insights(regional_stats, cv_data)
    
    all_results['regional_stats'] = regional_stats
    all_results['cv_data'] = cv_data
    
    # 2. Análisis de Precio por m²
    print("\n\n2. ANÁLISIS DE PRECIO POR M²")
    print("=" * 30)
    
    sqm_stats, premium_threshold = analyze_sqm_prices(df)
    comparison_df = create_ranking_comparison(regional_stats, sqm_stats)
    create_sqm_price_plots(df, sqm_stats, premium_threshold, comparison_df)
    print_sqm_insights(sqm_stats, comparison_df)
    
    all_results['sqm_stats'] = sqm_stats
    all_results['premium_threshold'] = premium_threshold
    all_results['comparison_df'] = comparison_df
    
    # 3. Análisis de Volumen
    print("\n\n3. ANÁLISIS DE VOLUMEN DE TRANSACCIONES")
    print("=" * 40)
    
    volume_stats, correlation, high_liquidity_threshold = analyze_transaction_volume(df, target)
    print_volume_summary(volume_stats, correlation, high_liquidity_threshold)
    create_volume_plots(volume_stats, correlation, high_liquidity_threshold)
    print_volume_insights(volume_stats, correlation)
    
    all_results['volume_stats'] = volume_stats
    all_results['correlation'] = correlation
    all_results['high_liquidity_threshold'] = high_liquidity_threshold
    
    # 4. Análisis Temporal (si está disponible)
    yearly_stats = None
    if 'date' in df.columns:
        print("\n\n4. ANÁLISIS TEMPORAL")
        print("=" * 20)
        
        yearly_stats = analyze_temporal_trends(df, target)
        create_temporal_plots(yearly_stats)
        print_temporal_insights(yearly_stats)
        all_results['yearly_stats'] = yearly_stats
    
    # 5. Análisis por Tipo de Propiedad
    print("\n\n5. ANÁLISIS POR TIPO DE PROPIEDAD")
    print("=" * 35)
    
    property_analysis = analyze_property_types(df, target)
    price_stats_formatted = format_property_type_stats(property_analysis[0])
    print_property_type_distribution(property_analysis[2])
    
    if include_visualizations:
        create_property_type_plots(df, target)
    
    property_significance = analyze_property_type_significance(df, target)
    
    all_results['property_analysis'] = property_analysis
    all_results['property_significance'] = property_significance
    
    # 6. Análisis de Comportamiento de Mercado
    print("\n\n6. ANÁLISIS DEL COMPORTAMIENTO DE MERCADO")
    print("=" * 45)
    
    market_behavior = analyze_market_behavior(df, target)
    if include_visualizations:
        create_market_behavior_plots(df, market_behavior, target)
    
    seasonal_patterns = analyze_seasonal_patterns(df, target)
    
    all_results['market_behavior'] = market_behavior
    all_results['seasonal_patterns'] = seasonal_patterns
    
    # 7. Segmentación de Mercado
    print("\n\n7. SEGMENTACIÓN DE MERCADO")
    print("=" * 30)
    
    market_segmentation = analyze_market_segmentation(df, target)
    if include_visualizations:
        create_market_segmentation_plots(df, market_segmentation, target)
        create_niche_analysis_plots(df, market_segmentation.get('niche_analysis', {}), target)
    
    print_segmentation_insights(market_segmentation, df)
    
    all_results['market_segmentation'] = market_segmentation
    
    # Resumen ejecutivo
    print("\n\n" + "=" * 60)
    print("RESUMEN EJECUTIVO")
    print("=" * 60)
    
    print(f"Dataset analizado: {len(df):,} transacciones")
    print(f"Regiones analizadas: {df['region'].nunique()}")
    print(f"Precio promedio nacional: {df[target].mean():,.0f} DKK")
    print(f"Precio/m² promedio nacional: {df['sqm_price'].mean():,.0f} DKK/m²")
    
    return all_results
    print(f"Región más cara: {regional_stats.index[0]}")
    print(f"Región con mayor volumen: {volume_stats.index[0]}")
    
    return {
        'regional_stats': regional_stats,
        'sqm_stats': sqm_stats,
        'volume_stats': volume_stats,
        'yearly_stats': yearly_stats,
        'comparison_df': comparison_df,
        'premium_threshold': premium_threshold,
        'correlation': correlation,
        'high_liquidity_threshold': high_liquidity_threshold,
        'cv_data': cv_data
    }


# =============================================================================
# FUNCIONES AUXILIARES PARA EXPORTACIÓN DE RESULTADOS
# =============================================================================

def export_results_to_csv(results, output_dir='results/tablas/'):
    """
    Exportar resultados del análisis a archivos CSV.
    
    Parameters:
    -----------
    results : dict
        Diccionario con resultados del análisis
    output_dir : str
        Directorio de salida para los archivos
    """
    import os
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Exportar cada resultado
    for key, value in results.items():
        if isinstance(value, pd.DataFrame) and value is not None:
            filename = f"{output_dir}descriptive_analysis_{key}.csv"
            value.to_csv(filename)
            print(f"Exportado: {filename}")


def generate_summary_report(results):
    """
    Generar reporte resumen del análisis descriptivo.
    
    Parameters:
    -----------
    results : dict
        Diccionario con resultados del análisis
        
    Returns:
    --------
    str
        Reporte en formato texto
    """
    report = []
    report.append("=" * 80)
    report.append("REPORTE RESUMEN - ANÁLISIS DESCRIPTIVO")
    report.append("=" * 80)
    
    if 'regional_stats' in results:
        regional = results['regional_stats']
        report.append(f"\n1. ANÁLISIS REGIONAL:")
        report.append(f"   • Total regiones: {len(regional)}")
        report.append(f"   • Región más cara: {regional.index[0]}")
        report.append(f"   • Precio promedio máximo: {regional.iloc[0]['Promedio']:,.0f} DKK")
    
    if 'sqm_stats' in results:
        sqm = results['sqm_stats']
        report.append(f"\n2. ANÁLISIS PRECIO/M²:")
        report.append(f"   • Región más eficiente: {sqm.index[0]}")
        report.append(f"   • Precio/m² máximo: {sqm.iloc[0]['Promedio_m2']:,.0f} DKK/m²")
        report.append(f"   • Regiones premium: {sqm['Es_Premium'].sum()}")
    
    if 'volume_stats' in results:
        volume = results['volume_stats']
        report.append(f"\n3. ANÁLISIS VOLUMEN:")
        report.append(f"   • Región líder: {volume.index[0]}")
        report.append(f"   • Total transacciones: {volume['Num_Transacciones'].sum():,}")
        report.append(f"   • Concentración top 10: {volume.head(10)['Participacion_Mercado'].sum():.1f}%")
    
    return "\n".join(report)

# =============================================================================
# ANÁLISIS POR TIPO DE PROPIEDAD (SECCIÓN 3)
# =============================================================================

def analyze_property_types(df, target='purchase_price'):
    """
    Analizar diferencias por tipo de propiedad.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    target : str
        Columna objetivo (precio)
        
    Returns:
    --------
    tuple
        (price_stats_by_type, physical_stats_by_type, regional_dist_by_type)
    """
    print("🏠 ANÁLISIS POR TIPO DE PROPIEDAD")
    print("=" * 50)
    
    # 3.1 Estadísticas básicas por tipo
    print("\n📊 3.1 Estadísticas de precios por tipo de propiedad:")
    price_stats_by_type = df.groupby('house_type')[target].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(0)
    price_stats_by_type['cv'] = (price_stats_by_type['std'] / price_stats_by_type['mean'] * 100).round(2)
    price_stats_by_type = price_stats_by_type.sort_values('median', ascending=False)
    
    # 3.2 Características físicas por tipo
    print("\n🏗️ 3.2 Características físicas por tipo:")
    physical_stats_by_type = df.groupby('house_type').agg({
        'sqm': ['mean', 'median', 'std'],
        'no_rooms': ['mean', 'median', 'std'],
        'sqm_price': ['mean', 'median', 'std']
    }).round(2)
    
    # Aplanar columnas multinivel
    physical_stats_by_type.columns = ['_'.join(col).strip() for col in physical_stats_by_type.columns]
    
    # 3.3 Distribución regional por tipo
    print("\n🌍 3.3 Distribución regional por tipo de propiedad:")
    regional_dist_by_type = pd.crosstab(df['region'], df['house_type'], normalize='columns') * 100
    regional_dist_by_type = regional_dist_by_type.round(1)
    
    return price_stats_by_type, physical_stats_by_type, regional_dist_by_type


def format_property_type_stats(price_stats_by_type):
    """
    Formatear estadísticas de tipo de propiedad para mostrar.
    
    Parameters:
    -----------
    price_stats_by_type : pd.DataFrame
        Estadísticas de precios por tipo
        
    Returns:
    --------
    pd.DataFrame
        Estadísticas formateadas
    """
    price_stats_formatted = price_stats_by_type.copy()
    for col in ['count', 'mean', 'median', 'std', 'min', 'max']:
        if col == 'count':
            price_stats_formatted[col] = price_stats_formatted[col].apply(lambda x: f"{x:,.0f}")
        else:
            price_stats_formatted[col] = price_stats_formatted[col].apply(lambda x: f"{x:,.0f} DKK")
    price_stats_formatted['cv'] = price_stats_formatted['cv'].apply(lambda x: f"{x:.1f}%")
    
    return price_stats_formatted


def print_property_type_distribution(regional_dist_by_type):
    """
    Imprimir distribución regional por tipo de propiedad.
    
    Parameters:
    -----------
    regional_dist_by_type : pd.DataFrame
        Distribución regional por tipo
    """
    print("🔝 Top 3 regiones por concentración de cada tipo:")
    for house_type in regional_dist_by_type.columns:
        top_regions_for_type = regional_dist_by_type[house_type].nlargest(3)
        print(f"\n{house_type}:")
        for region, pct in top_regions_for_type.items():
            print(f"  • {region}: {pct:.1f}%")


def create_property_type_plots(df, target='purchase_price'):
    """
    Crear visualizaciones por tipo de propiedad.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    target : str
        Columna objetivo (precio)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análisis por Tipo de Propiedad', fontsize=16, fontweight='bold', y=0.98)

    # 1. Boxplot de precios por tipo
    sns.boxplot(data=df, x='house_type', y=target, ax=axes[0,0])
    axes[0,0].set_title('Distribución de Precios por Tipo de Propiedad')
    axes[0,0].set_xlabel('Tipo de Propiedad')
    axes[0,0].set_ylabel('Precio (DKK)')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].set_ylim(0, 10000000)  # Limitar para mejor visualización

    # 2. Precio por m² por tipo
    sns.boxplot(data=df, x='house_type', y='sqm_price', ax=axes[0,1])
    axes[0,1].set_title('Precio por m² por Tipo de Propiedad')
    axes[0,1].set_xlabel('Tipo de Propiedad')
    axes[0,1].set_ylabel('Precio por m² (DKK)')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].set_ylim(0, 80000)  # Limitar para mejor visualización

    # 3. Tamaño promedio por tipo
    avg_size_by_type = df.groupby('house_type')['sqm'].mean().sort_values(ascending=True)
    avg_size_by_type.plot(kind='barh', ax=axes[1,0], color='lightcoral')
    axes[1,0].set_title('Tamaño Promedio por Tipo de Propiedad')
    axes[1,0].set_xlabel('Superficie (m²)')

    # 4. Volumen de transacciones por tipo
    transaction_volume_by_type = df['house_type'].value_counts()
    transaction_volume_by_type.plot(kind='pie', ax=axes[1,1], autopct='%1.1f%%')
    axes[1,1].set_title('Volumen de Transacciones por Tipo')
    axes[1,1].set_ylabel('')

    plt.tight_layout()
    plt.show()


def analyze_property_type_significance(df, target='purchase_price'):
    """
    Analizar significancia estadística entre tipos de propiedad.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    target : str
        Columna objetivo (precio)
        
    Returns:
    --------
    dict
        Resultados de tests estadísticos
    """
    from scipy.stats import f_oneway
    
    print("\n3.4 Tests de significancia estadística entre tipos de propiedad:")
    
    # ANOVA para comparar precios entre tipos
    house_types = df['house_type'].unique()
    price_groups = [df[df['house_type'] == ht][target] for ht in house_types]
    f_stat, p_value = f_oneway(*price_groups)

    print(f"ANOVA - Diferencias de precio entre tipos de propiedad:")
    print(f"F-estadístico: {f_stat:.2f}")
    print(f"p-valor: {p_value:.2e}")
    print(f"Interpretación: {'Hay diferencias significativas' if p_value < 0.05 else 'No hay diferencias significativas'}")

    # Correlación entre características físicas y precio por tipo
    print("\n3.5 Correlaciones precio vs características por tipo:")
    correlations = {}
    for house_type in house_types:
        subset = df[df['house_type'] == house_type]
        corr_sqm = subset[target].corr(subset['sqm'])
        corr_rooms = subset[target].corr(subset['no_rooms'])
        correlations[house_type] = {
            'size_correlation': corr_sqm,
            'rooms_correlation': corr_rooms
        }
        print(f"{house_type}:")
        print(f"  Correlación precio-tamaño: {corr_sqm:.3f}")
        print(f"  Correlación precio-habitaciones: {corr_rooms:.3f}")
    
    return {
        'anova_f_stat': f_stat,
        'anova_p_value': p_value,
        'correlations': correlations
    }


# =============================================================================
# ANÁLISIS DEL COMPORTAMIENTO DE MERCADO (SECCIÓN 4)
# =============================================================================

def analyze_market_behavior(df, target='purchase_price'):
    """
    Analizar comportamiento del mercado inmobiliario.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    target : str
        Columna objetivo (precio)
        
    Returns:
    --------
    dict
        Resultados del análisis de mercado
    """
    print("=== ANÁLISIS DEL COMPORTAMIENTO DE MERCADO ===")
    
    results = {}
    
    # 4.1 Verificar columnas disponibles
    print("4.1 Columnas disponibles relacionadas con mercado:")
    market_columns = [col for col in df.columns if any(keyword in col.lower() 
                     for keyword in ['sales', 'type', 'change', 'offer', 'quarter'])]
    print("Columnas encontradas:", market_columns)
    results['available_columns'] = market_columns
    
    # 4.2 Análisis por sales_type si está disponible
    if 'sales_type' in df.columns:
        print("\n4.1 Análisis por tipo de venta (sales_type):")
        sales_analysis = df.groupby('sales_type').agg({
            target: ['count', 'mean', 'median', 'std'],
            'sqm_price': ['mean', 'median']
        }).round(0)
        
        # Aplanar columnas
        sales_analysis.columns = ['_'.join(col) for col in sales_analysis.columns]
        results['sales_analysis'] = sales_analysis
    else:
        print("\n4.1 sales_type no disponible en el dataset")
        results['sales_analysis'] = None
    
    # 4.3 Análisis de cambio oferta-compra
    change_col = None
    for col in df.columns:
        if 'change' in col.lower() and ('offer' in col.lower() or 'purchase' in col.lower()):
            change_col = col
            break
    
    if change_col:
        print(f"\n4.2 Análisis de variación precio oferta vs compra ({change_col}):")
        change_stats = df[change_col].describe()
        results['change_analysis'] = {
            'column': change_col,
            'stats': change_stats,
            'categories': analyze_price_change_categories(df, change_col)
        }
    else:
        print("\n4.2 Columna de cambio oferta-compra no disponible")
        results['change_analysis'] = None
    
    # 4.4 Análisis temporal
    temporal_results = analyze_temporal_patterns(df, target)
    results.update(temporal_results)
    
    return results


def analyze_price_change_categories(df, change_col):
    """
    Categorizar cambios entre oferta y compra.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    change_col : str
        Columna de cambio de precio
        
    Returns:
    --------
    pd.Series
        Distribución de categorías
    """
    df_temp = df.copy()
    df_temp['change_category'] = pd.cut(
        df_temp[change_col],
        bins=[-float('inf'), -5, -1, 1, 5, float('inf')],
        labels=['Descuento >5%', 'Descuento 1-5%', 'Sin cambio ±1%', 'Premium 1-5%', 'Premium >5%']
    )
    change_dist = df_temp['change_category'].value_counts()
    
    print(f"Distribución de cambios:")
    for cat, count in change_dist.items():
        pct = count / len(df_temp) * 100
        print(f"  {cat}: {count:,} ({pct:.1f}%)")
    
    return change_dist


def analyze_temporal_patterns(df, target='purchase_price'):
    """
    Analizar patrones temporales usando fecha.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    target : str
        Columna objetivo (precio)
        
    Returns:
    --------
    dict
        Resultados del análisis temporal
    """
    results = {}
    
    if 'date' in df.columns:
        print("\n4.4 Análisis estacional usando la fecha:")
        
        df_temp = df.copy()
        # Crear variables temporales
        df_temp['year'] = pd.to_datetime(df_temp['date']).dt.year
        df_temp['month'] = pd.to_datetime(df_temp['date']).dt.month
        df_temp['quarter_from_date'] = pd.to_datetime(df_temp['date']).dt.quarter
        
        # Análisis por trimestre
        quarterly_from_date = df_temp.groupby('quarter_from_date').agg({
            target: ['count', 'mean', 'median'],
            'sqm_price': ['mean', 'median']
        }).round(0)
        quarterly_from_date.columns = ['_'.join(col) for col in quarterly_from_date.columns]
        
        # Análisis por mes
        monthly_stats = df_temp.groupby('month')[target].agg(['count', 'mean', 'median']).round(0)
        
        results['quarterly_analysis'] = quarterly_from_date
        results['monthly_analysis'] = monthly_stats
        results['temporal_variables'] = ['year', 'month', 'quarter_from_date']
        
        print("Estadísticas por trimestre (derivado de fecha):")
        # No mostrar aquí para evitar duplicación
        
    else:
        print("\n4.4 Análisis temporal limitado - columna date no disponible")
        results['quarterly_analysis'] = None
        results['monthly_analysis'] = None
    
    return results


def create_market_behavior_plots(df, results, target='purchase_price'):
    """
    Crear visualizaciones del comportamiento de mercado.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    results : dict
        Resultados del análisis de mercado
    target : str
        Columna objetivo (precio)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análisis del Comportamiento de Mercado', fontsize=16, fontweight='bold', y=0.98)

    # 1. Análisis por tipo de venta
    if results['sales_analysis'] is not None:
        # Filtrar tipos de venta con más de 1000 transacciones
        major_sales_types = df['sales_type'].value_counts()
        major_types = major_sales_types[major_sales_types > 1000].index
        df_major_sales = df[df['sales_type'].isin(major_types)]
        
        sns.boxplot(data=df_major_sales, x='sales_type', y=target, ax=axes[0,0])
        axes[0,0].set_title('Precios por Tipo de Venta')
        axes[0,0].set_xlabel('Tipo de Venta')
        axes[0,0].set_ylabel('Precio (DKK)')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].set_ylim(0, 8000000)
    else:
        axes[0,0].text(0.5, 0.5, 'Datos de sales_type\nno disponibles', 
                      ha='center', va='center', transform=axes[0,0].transAxes)
        axes[0,0].set_title('Tipo de Venta - No Disponible')

    # 2. Distribución de cambios oferta vs compra
    if results['change_analysis'] is not None:
        change_col = results['change_analysis']['column']
        change_filtered = df[change_col][(df[change_col] >= -25) & (df[change_col] <= 25)]
        change_filtered.hist(bins=50, ax=axes[0,1], alpha=0.7, color='skyblue')
        axes[0,1].axvline(0, color='red', linestyle='--', label='Sin cambio')
        axes[0,1].axvline(change_filtered.mean(), color='orange', linestyle='--', 
                         label=f'Media: {change_filtered.mean():.1f}%')
        axes[0,1].set_title('Distribución de Cambios Oferta vs Compra')
        axes[0,1].set_xlabel('% Cambio')
        axes[0,1].set_ylabel('Frecuencia')
        axes[0,1].legend()
    else:
        axes[0,1].text(0.5, 0.5, 'Datos de cambio\noferta-compra\nno disponibles', 
                      ha='center', va='center', transform=axes[0,1].transAxes)
        axes[0,1].set_title('Cambio Oferta-Compra - No Disponible')

    # 3. Análisis por trimestre
    if results['quarterly_analysis'] is not None and 'month' in df.columns:
        df_temp = df.copy()
        df_temp['quarter_from_date'] = pd.to_datetime(df_temp['date']).dt.quarter
        quarterly_prices = df_temp.groupby('quarter_from_date')[target].median()
        quarterly_prices.plot(kind='bar', ax=axes[1,0], color='lightgreen')
        axes[1,0].set_title('Precio Mediano por Trimestre')
        axes[1,0].set_xlabel('Trimestre')
        axes[1,0].set_ylabel('Precio Mediano (DKK)')
        axes[1,0].tick_params(axis='x', rotation=0)
    else:
        axes[1,0].text(0.5, 0.5, 'Datos de trimestre\nno disponibles', 
                      ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].set_title('Análisis Trimestral - No Disponible')

    # 4. Volumen por trimestre
    if results['quarterly_analysis'] is not None and 'month' in df.columns:
        df_temp = df.copy()
        df_temp['quarter_from_date'] = pd.to_datetime(df_temp['date']).dt.quarter
        quarterly_volume = df_temp['quarter_from_date'].value_counts().sort_index()
        quarterly_volume.plot(kind='bar', ax=axes[1,1], color='orange')
        axes[1,1].set_title('Volumen de Transacciones por Trimestre')
        axes[1,1].set_xlabel('Trimestre')
        axes[1,1].set_ylabel('Número de Transacciones')
        axes[1,1].tick_params(axis='x', rotation=0)
    else:
        axes[1,1].text(0.5, 0.5, 'Datos de trimestre\nno disponibles', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Volumen Trimestral - No Disponible')

    plt.tight_layout()
    plt.show()


def print_volume_insights(volume_stats, correlation):
    """
    Imprimir insights del análisis de volumen.
    
    Parameters:
    -----------
    volume_stats : pd.DataFrame
        Estadísticas de volumen
    correlation : float
        Correlación volumen-precio
    """
    print("\nINSIGHTS CLAVE - VOLUMEN DE TRANSACCIONES")
    print("=" * 45)
    print(f"Región líder: {volume_stats.index[0]} ({volume_stats.iloc[0]['Num_Transacciones']:,} trans.)")
    print(f"Participación del líder: {volume_stats.iloc[0]['Participacion_Mercado']:.1f}%")
    print(f"Mercados principales: {(volume_stats['Tipo_Mercado'] == 'Principal').sum()} regiones")
    print(f"Concentración top 5: {volume_stats.head(5)['Participacion_Mercado'].sum():.1f}%")
    print(f"Correlación volumen-precio: {'Positiva' if correlation > 0 else 'Negativa'} ({abs(correlation):.3f})")


# =============================================================================
# ANÁLISIS TEMPORAL
# =============================================================================

def analyze_temporal_trends(df):
    """
    Análisis de tendencias temporales de precios.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos de transacciones
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con estadísticas anuales
    """
    # Crear columnas de fecha
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['month'] = pd.to_datetime(df['date']).dt.month
    
    # Estadísticas anuales
    yearly_stats = df.groupby('year').agg({
        'purchase_price': ['count', 'mean', 'median', 'std'],
        'sqm_price': ['mean', 'median']
    }).round(0)
    
    yearly_stats.columns = ['Transacciones', 'Precio_Promedio', 'Precio_Mediana', 'Precio_Std',
                           'Precio_m2_Promedio', 'Precio_m2_Mediana']
    
    # Calcular tasas de crecimiento anual
    yearly_stats['Crecimiento_Precio'] = yearly_stats['Precio_Promedio'].pct_change() * 100
    yearly_stats['Crecimiento_m2'] = yearly_stats['Precio_m2_Promedio'].pct_change() * 100
    
    return yearly_stats


def create_temporal_plots(yearly_stats, figsize=(16, 10)):
    """
    Crear visualizaciones de tendencias temporales.
    
    Parameters:
    -----------
    yearly_stats : pd.DataFrame
        Estadísticas anuales
    figsize : tuple
        Tamaño de la figura
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Evolución Temporal del Mercado Inmobiliario Danés', fontsize=16, fontweight='bold')
    
    years = yearly_stats.index
    
    # 1. Evolución de precios promedio
    ax1.plot(years, yearly_stats['Precio_Promedio'], marker='o', linewidth=2, markersize=4)
    ax1.set_title('Evolución del Precio Promedio')
    ax1.set_xlabel('Año')
    ax1.set_ylabel('Precio Promedio (DKK)')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Evolución de precio por m²
    ax2.plot(years, yearly_stats['Precio_m2_Promedio'], marker='s', linewidth=2, 
             markersize=4, color='orange')
    ax2.set_title('Evolución del Precio por m²')
    ax2.set_xlabel('Año')
    ax2.set_ylabel('Precio por m² (DKK)')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Volumen de transacciones
    ax3.bar(years, yearly_stats['Transacciones'], alpha=0.7, color='green')
    ax3.set_title('Volumen de Transacciones por Año')
    ax3.set_xlabel('Año')
    ax3.set_ylabel('Número de Transacciones')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Tasas de crecimiento
    ax4.plot(years[1:], yearly_stats['Crecimiento_Precio'].dropna(), marker='o', 
             linewidth=2, label='Precio Total', alpha=0.8)
    ax4.plot(years[1:], yearly_stats['Crecimiento_m2'].dropna(), marker='s', 
             linewidth=2, label='Precio/m²', alpha=0.8)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax4.set_title('Tasas de Crecimiento Anual')
    ax4.set_xlabel('Año')
    ax4.set_ylabel('Crecimiento (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def print_temporal_insights(yearly_stats):
    """
    Imprimir insights del análisis temporal.
    
    Parameters:
    -----------
    yearly_stats : pd.DataFrame
        Estadísticas anuales
    """
    print("INSIGHTS TEMPORALES")
    print("=" * 25)
    
    # Períodos de mayor crecimiento
    max_growth_year = yearly_stats['Crecimiento_Precio'].idxmax()
    max_growth_rate = yearly_stats['Crecimiento_Precio'].max()
    
    # Períodos de mayor declive
    min_growth_year = yearly_stats['Crecimiento_Precio'].idxmin()
    min_growth_rate = yearly_stats['Crecimiento_Precio'].min()
    
    print(f"Mayor crecimiento: {max_growth_year} ({max_growth_rate:.1f}%)")
    print(f"Mayor declive: {min_growth_year} ({min_growth_rate:.1f}%)")
    print(f"Crecimiento promedio anual: {yearly_stats['Crecimiento_Precio'].mean():.1f}%")
    print(f"Período analizado: {yearly_stats.index.min()} - {yearly_stats.index.max()}")


# =============================================================================
# ANÁLISIS DE SEGMENTACIÓN DE MERCADO (SECCIÓN 5)
# =============================================================================

def analyze_market_segmentation(df, target='purchase_price'):
    """
    Analizar segmentación del mercado inmobiliario.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    target : str
        Columna objetivo (precio)
        
    Returns:
    --------
    dict
        Resultados de la segmentación
    """
    print("=== SEGMENTACIÓN DE MERCADO ===")
    
    results = {}
    
    # 5.1 Segmentación por precio
    price_segments = analyze_price_segmentation(df, target)
    results['price_segmentation'] = price_segments
    
    # 5.2 Segmentación por antigüedad
    if 'year_build' in df.columns:
        age_segments = analyze_age_segmentation(df, target)
        results['age_segmentation'] = age_segments
    else:
        print("Columna year_build no disponible")
        results['age_segmentation'] = None
    
    # 5.3 Análisis de nichos
    niche_analysis = analyze_niche_markets(df, target)
    results['niche_analysis'] = niche_analysis
    
    # 5.4 Segmentación urbano vs rural
    urban_rural_analysis = analyze_urban_rural_segmentation(df, target)
    results['urban_rural'] = urban_rural_analysis
    
    return results


def analyze_price_segmentation(df, target='purchase_price'):
    """
    Segmentación por nivel de precios.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    target : str
        Columna objetivo (precio)
        
    Returns:
    --------
    dict
        Resultados de segmentación por precio
    """
    print("\n5.1 Segmentación premium vs económico:")

    # Definir umbrales
    q25 = df[target].quantile(0.25)
    q75 = df[target].quantile(0.75) 
    q90 = df[target].quantile(0.90)

    print(f"Umbrales de segmentación:")
    print(f"- Q25 (Económico): {q25:,.0f} DKK")
    print(f"- Q75 (Alto): {q75:,.0f} DKK")
    print(f"- Q90 (Premium): {q90:,.0f} DKK")

    df_temp = df.copy()
    df_temp['price_segment'] = pd.cut(
        df_temp[target],
        bins=[0, q25, q75, q90, float('inf')],
        labels=['Económico', 'Medio', 'Alto', 'Premium']
    )

    segment_stats = df_temp.groupby('price_segment').agg({
        target: ['count', 'mean', 'median', 'min', 'max'],
        'sqm': ['mean', 'median'],
        'no_rooms': ['mean', 'median'],
        'sqm_price': ['mean', 'median']
    }).round(0)

    segment_stats.columns = ['_'.join(col) for col in segment_stats.columns]
    
    return {
        'thresholds': {'q25': q25, 'q75': q75, 'q90': q90},
        'segment_stats': segment_stats,
        'df_with_segments': df_temp
    }


def analyze_age_segmentation(df, target='purchase_price'):
    """
    Segmentación por antigüedad de la propiedad.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    target : str
        Columna objetivo (precio)
        
    Returns:
    --------
    dict
        Resultados de segmentación por antigüedad
    """
    print("\n5.2 Segmentación por antigüedad de la propiedad:")
    
    current_year = 2024
    df_temp = df.copy()
    df_temp['property_age'] = current_year - df_temp['year_build']
    
    # Filtrar años válidos
    df_temp = df_temp[df_temp['year_build'] > 1800]
    
    df_temp['age_category'] = pd.cut(
        df_temp['property_age'],
        bins=[0, 10, 25, 50, 100, float('inf')],
        labels=['Nueva (0-10 años)', 'Moderna (11-25 años)', 
               'Madura (26-50 años)', 'Antigua (51-100 años)', 'Histórica (>100 años)']
    )
    
    age_stats = df_temp.groupby('age_category').agg({
        target: ['count', 'mean', 'median'],
        'sqm_price': ['mean', 'median'],
        'property_age': ['mean', 'median']
    }).round(0)
    
    age_stats.columns = ['_'.join(col) for col in age_stats.columns]
    
    return {
        'age_stats': age_stats,
        'df_with_age': df_temp
    }


def analyze_niche_markets(df, target='purchase_price'):
    """
    Analizar mercados de nicho específicos.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    target : str
        Columna objetivo (precio)
        
    Returns:
    --------
    dict
        Análisis de mercados de nicho
    """
    print("\n5.3 Análisis de mercados de nicho:")
    
    niche_types = ['Farm', 'Summerhouse']
    available_niches = [nt for nt in niche_types if nt in df['house_type'].values]
    
    niche_analysis = {}
    
    if available_niches:
        for niche in available_niches:
            niche_data = df[df['house_type'] == niche]
            
            print(f"\n--- Análisis de {niche} ---")
            print(f"Número de propiedades: {len(niche_data):,}")
            print(f"Precio promedio: {niche_data[target].mean():,.0f} DKK")
            print(f"Precio mediano: {niche_data[target].median():,.0f} DKK")
            print(f"Tamaño promedio: {niche_data['sqm'].mean():.0f} m²")
            print(f"Precio/m² promedio: {niche_data['sqm_price'].mean():,.0f} DKK/m²")
            
            # Top regiones para este nicho
            top_regions = niche_data['region'].value_counts().head(5)
            print(f"Top 5 regiones:")
            for region, count in top_regions.items():
                pct = count / len(niche_data) * 100
                print(f"  {region}: {count} ({pct:.1f}%)")
            
            niche_analysis[niche] = {
                'data': niche_data,
                'count': len(niche_data),
                'avg_price': niche_data[target].mean(),
                'median_price': niche_data[target].median(),
                'avg_size': niche_data['sqm'].mean(),
                'avg_sqm_price': niche_data['sqm_price'].mean(),
                'top_regions': top_regions
            }
            
            # Análisis estacional para nichos relevantes
            if 'date' in df.columns and niche == 'Summerhouse':
                seasonal_analysis = analyze_niche_seasonality(niche_data, niche)
                niche_analysis[niche]['seasonality'] = seasonal_analysis
    
    return niche_analysis


def analyze_niche_seasonality(niche_data, niche_name):
    """
    Analizar estacionalidad para mercados de nicho.
    
    Parameters:
    -----------
    niche_data : pd.DataFrame
        Datos del nicho específico
    niche_name : str
        Nombre del nicho
        
    Returns:
    --------
    dict
        Análisis de estacionalidad
    """
    print(f"Estacionalidad de {niche_name}:")
    
    niche_temp = niche_data.copy()
    niche_temp['month'] = pd.to_datetime(niche_temp['date']).dt.month
    seasonal_volume = niche_temp['month'].value_counts().sort_index()
    peak_months = seasonal_volume.nlargest(3)
    
    print(f"Meses pico de transacciones:")
    for month, count in peak_months.items():
        print(f"  Mes {month}: {count} transacciones")
    
    return {
        'seasonal_volume': seasonal_volume,
        'peak_months': peak_months
    }


def analyze_urban_rural_segmentation(df, target='purchase_price'):
    """
    Segmentación urbano vs rural.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    target : str
        Columna objetivo (precio)
        
    Returns:
    --------
    dict
        Análisis urbano vs rural
    """
    print("\n5.4 Aproximación de segmentación urbano vs rural:")
    print("Clasificación aproximada basada en regiones:")

    # Clasificación aproximada (esto podría refinarse con más datos geográficos)
    urban_regions = ['Zealand']  # Zealand incluye Copenhague y área metropolitana
    rural_regions = ['Jutland', 'Fyn & islands', 'Bornholm']

    df_temp = df.copy()
    df_temp['area_type'] = df_temp['region'].apply(
        lambda x: 'Urbano' if x in urban_regions else 'Rural'
    )

    urban_rural_stats = df_temp.groupby('area_type').agg({
        target: ['count', 'mean', 'median'],
        'sqm_price': ['mean', 'median'],
        'sqm': ['mean', 'median']
    }).round(0)

    urban_rural_stats.columns = ['_'.join(col) for col in urban_rural_stats.columns]

    print("\nNota: La clasificación urbano/rural es aproximada basada en regiones.")
    print("Zealand (incluye Copenhague) se considera urbano, el resto rural.")
    
    return {
        'urban_regions': urban_regions,
        'rural_regions': rural_regions,
        'stats': urban_rural_stats,
        'df_with_area_type': df_temp
    }


def format_segmentation_stats(segment_stats):
    """
    Formatear estadísticas de segmentación para mostrar.
    
    Parameters:
    -----------
    segment_stats : pd.DataFrame
        Estadísticas de segmentación
        
    Returns:
    --------
    pd.DataFrame
        Estadísticas formateadas
    """
    segment_formatted = segment_stats.copy()
    for col in segment_formatted.columns:
        if 'count' in col:
            segment_formatted[col] = segment_formatted[col].apply(lambda x: f"{x:,.0f}")
        elif any(price_col in col for price_col in ['purchase_price', 'sqm_price']):
            segment_formatted[col] = segment_formatted[col].apply(lambda x: f"{x:,.0f} DKK")
        elif any(metric in col for metric in ['sqm_', 'no_rooms']):
            segment_formatted[col] = segment_formatted[col].apply(lambda x: f"{x:.1f}")
        elif 'property_age' in col:
            segment_formatted[col] = segment_formatted[col].apply(lambda x: f"{x:.0f} años")

    return segment_formatted


def create_market_segmentation_plots(df, segmentation_results, target='purchase_price'):
    """
    Crear visualizaciones de segmentación de mercado.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    segmentation_results : dict
        Resultados de la segmentación
    target : str
        Columna objetivo (precio)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Segmentación de Mercado', fontsize=16, fontweight='bold', y=0.98)

    # 1. Segmentación por precio
    if 'price_segmentation' in segmentation_results:
        df_temp = segmentation_results['price_segmentation']['df_with_segments']
        segment_counts = df_temp['price_segment'].value_counts()
        segment_counts.plot(kind='pie', ax=axes[0,0], autopct='%1.1f%%', 
                           colors=['lightcoral', 'lightblue', 'lightgreen', 'gold'])
        axes[0,0].set_title('Distribución por Segmento de Precio')
        axes[0,0].set_ylabel('')

        # Precio por segmento
        sns.boxplot(data=df_temp, x='price_segment', y=target, ax=axes[0,1])
        axes[0,1].set_title('Distribución de Precios por Segmento')
        axes[0,1].set_xlabel('Segmento')
        axes[0,1].set_ylabel('Precio (DKK)')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].set_ylim(0, 8000000)
    else:
        axes[0,0].text(0.5, 0.5, 'Segmentación\nde precio\nno disponible', 
                      ha='center', va='center', transform=axes[0,0].transAxes)
        axes[0,1].text(0.5, 0.5, 'Segmentación\nde precio\nno disponible', 
                      ha='center', va='center', transform=axes[0,1].transAxes)

    # 3. Segmentación por antigüedad
    if segmentation_results.get('age_segmentation') is not None:
        df_age = segmentation_results['age_segmentation']['df_with_age']
        age_counts = df_age['age_category'].value_counts()
        age_counts.plot(kind='bar', ax=axes[1,0], color='lightcoral')
        axes[1,0].set_title('Distribución por Antigüedad')
        axes[1,0].set_xlabel('Categoría de Antigüedad')
        axes[1,0].set_ylabel('Número de Propiedades')
        axes[1,0].tick_params(axis='x', rotation=45)
    else:
        axes[1,0].text(0.5, 0.5, 'Datos de año de\nconstrucción\nno disponibles', 
                      ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].set_title('Antigüedad - No Disponible')

    # 4. Urbano vs Rural
    if 'urban_rural' in segmentation_results:
        df_urban_rural = segmentation_results['urban_rural']['df_with_area_type']
        urban_rural_counts = df_urban_rural['area_type'].value_counts()
        urban_rural_counts.plot(kind='bar', ax=axes[1,1], color=['skyblue', 'lightgreen'])
        axes[1,1].set_title('Distribución Urbano vs Rural')
        axes[1,1].set_xlabel('Tipo de Área')
        axes[1,1].set_ylabel('Número de Propiedades')
        axes[1,1].tick_params(axis='x', rotation=0)
    else:
        axes[1,1].text(0.5, 0.5, 'Clasificación\nurbano/rural\nno disponible', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Urbano vs Rural - No Disponible')

    plt.tight_layout()
    plt.show()


def create_niche_analysis_plots(df, niche_analysis, target='purchase_price'):
    """
    Crear visualizaciones para análisis de nichos.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    niche_analysis : dict
        Análisis de mercados de nicho
    target : str
        Columna objetivo (precio)
    """
    import matplotlib.pyplot as plt
    
    print("\n5.5 Análisis visual de mercados de nicho:")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Comparación de precios por m² entre tipos
    niche_sqm_prices = df.groupby('house_type')['sqm_price'].median().sort_values(ascending=False)
    niche_sqm_prices.plot(kind='bar', ax=axes[0], color='lightsteelblue')
    axes[0].set_title('Precio Mediano por m² por Tipo de Propiedad')
    axes[0].set_xlabel('Tipo de Propiedad')
    axes[0].set_ylabel('Precio por m² (DKK)')
    axes[0].tick_params(axis='x', rotation=45)

    # Análisis de estacionalidad para Summerhouse
    if 'date' in df.columns and 'Summerhouse' in niche_analysis:
        summerhouse_data = niche_analysis['Summerhouse']['data']
        if 'seasonality' in niche_analysis['Summerhouse']:
            seasonal_volume = niche_analysis['Summerhouse']['seasonality']['seasonal_volume']
            seasonal_volume.plot(kind='line', ax=axes[1], color='orange', marker='o')
            axes[1].set_title('Estacionalidad - Transacciones de Casas de Verano')
            axes[1].set_xlabel('Mes')
            axes[1].set_ylabel('Número de Transacciones')
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'Análisis estacional\nno disponible', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Estacionalidad Summerhouse - No Disponible')
    else:
        axes[1].text(0.5, 0.5, 'Datos de Summerhouse\no fecha no disponibles', 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Estacionalidad - No Disponible')

    plt.tight_layout()
    plt.show()


def print_segmentation_insights(segmentation_results, df):
    """
    Imprimir insights de segmentación.
    
    Parameters:
    -----------
    segmentation_results : dict
        Resultados de la segmentación
    df : pd.DataFrame
        DataFrame original
    """
    print("\n5.6 Resumen de insights de segmentación:")
    print("="*60)

    # Insights de precio
    if 'price_segmentation' in segmentation_results:
        df_temp = segmentation_results['price_segmentation']['df_with_segments']
        print("SEGMENTACIÓN POR PRECIO:")
        for segment in ['Económico', 'Medio', 'Alto', 'Premium']:
            if segment in df_temp['price_segment'].values:
                segment_data = df_temp[df_temp['price_segment'] == segment]
                pct_total = len(segment_data) / len(df_temp) * 100
                print(f"- {segment}: {len(segment_data):,} propiedades ({pct_total:.1f}%)")

    # Insights urbano/rural
    if 'urban_rural' in segmentation_results:
        df_urban_rural = segmentation_results['urban_rural']['df_with_area_type']
        print("\nSEGMENTACIÓN URBANO/RURAL:")
        urban_data = df_urban_rural[df_urban_rural['area_type'] == 'Urbano']
        rural_data = df_urban_rural[df_urban_rural['area_type'] == 'Rural']
        
        if len(urban_data) > 0 and len(rural_data) > 0:
            urban_premium = urban_data['purchase_price'].median() / rural_data['purchase_price'].median()
            price_diff = urban_data['sqm_price'].median() - rural_data['sqm_price'].median()
            print(f"- Premium urbano: {urban_premium:.1f}x más caro que rural")
            print(f"- Diferencia precio/m²: {price_diff:,.0f} DKK/m²")

    # Insights de nicho
    if 'niche_analysis' in segmentation_results:
        niche_analysis = segmentation_results['niche_analysis']
        if niche_analysis:
            print("\nMERCADOS DE NICHO:")
            regular_data = df[df['house_type'].isin(['Villa', 'Apartment', 'Townhouse'])]
            
            for niche_name, niche_data in niche_analysis.items():
                if 'avg_price' in niche_data and len(regular_data) > 0:
                    size_ratio = niche_data['avg_size'] / regular_data['sqm'].mean()
                    price_ratio = niche_data['median_price'] / regular_data['purchase_price'].median()
                    print(f"- {niche_name}: {size_ratio:.1f}x el tamaño promedio, {price_ratio:.2f}x el precio mediano")

    print("="*60)
