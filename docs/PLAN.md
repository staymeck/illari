# Illari — Laboratorio de análisis y trading cuantitativo (Fase 1)

## Contexto

Este proyecto (`illari`) venía desarrollándose en una sesión anterior: había un plan
acordado, un fetcher de datos de Binance, backtests corriendo y un reporte inicial
(`resources/report.md`, con solo 2 escenarios de muestra muy chica: 18 y 7 operaciones).
La PC se formateó y se perdió todo ese trabajo — el repositorio ni siquiera tenía
`git init`. Lo único que sobrevivió son los dos PDF y el reporte viejo en `resources/`.

Este plan reconstruye desde cero, por escrito y versionado en git, todo lo que se
había acordado en la conversación previa, para no depender de la memoria de una sesión
de chat. El objetivo de esta primera fase **no es programar todavía** — es dejar
documentado el diseño completo del laboratorio para que quede confirmado antes de
escribir código.

**Ya hecho en esta sesión:** `git init`, rama `main`, commit inicial con `resources/`.

## Objetivo del proyecto

Construir un **laboratorio de backtesting** que evalúe, con reglas objetivas y
medibles (no una caja negra de IA), qué tan bien habría funcionado una estrategia de
trading basada en:

- Estructura de mercado (soportes/resistencias, máximos/mínimos, rupturas)
- Geometría analítica sobre velas (líneas de tendencia, canales, retrocesos de Fibonacci
  como zonas de confluencia — no como señal aislada)
- Patrones de vela combinados con contexto (no patrones sueltos)
- Presión compradora/vendedora (volumen, y funding rate como proxy de sesgo del mercado)
- La hora/sesión de entrada al mercado, como variable de análisis

Filosofía explícita (ya acordada): no se busca "predecir" el mercado de forma
determinista. Se busca medir probabilidades con reglas objetivas y backtesting
riguroso — la mayoría de bots fallan por saltarse el rigor de esta fase, no por la
estrategia en sí.

### Las 3 fases del proyecto (visión completa, foco actual en Fase 1)

1. **Laboratorio histórico (foco de este plan):** correr la estrategia sobre datos
   pasados de varios mercados/escenarios y medir resultados (retorno, drawdown,
   win-rate, profit factor, desglose por hora/sesión).
2. **Señales en tiempo actual (a futuro):** mismo motor de análisis corriendo sobre
   datos en vivo, generando un "snapshot" de la condición actual del mercado y su
   probabilidad histórica asociada — sin ejecutar nada todavía.
3. **Ejecución automática (a futuro, condicionada):** solo si las Fases 1 y 2
   muestran resultados consistentes, conectar a un exchange para operar solo.

## Alcance de la Fase 1

### Mercados (todos en Binance, vía la misma API — sin infraestructura nueva)

| Símbolo | Rol |
|---|---|
| BTC/USDT | Referencia base, cripto "blue chip" |
| ETH/USDT | Blue chip con dinámica propia (DeFi/contratos) |
| PAXG/USDT | Oro tokenizado — único activo no-cripto real del set, sin sumar fuente de datos nueva |
| SOL/USDT o BNB/USDT | Altcoin de mayor beta/volatilidad |
| DOGE/USDT | Guiada por sentimiento/redes, no por fundamentos — dinámica genuinamente distinta |

**Advertencia a tener en cuenta al implementar:** no todos cotizan desde la misma
fecha en Binance (p. ej. SOL se lista después que BTC/ETH). Antes de fijar fechas de
escenario hay que chequear la fecha real de listado de cada símbolo contra la API, y
ajustar la ventana por activo en vez de forzar el mismo rango para los 5.

### Contextos temporales (multi-timeframe, visión global del mercado)

Se corre el mismo análisis en tres marcos de tiempo para no perder la vista global:

- **5 minutos** — entradas/timing fino
- **1 hora** — contexto de tendencia intermedia
- **1 día** — contexto de tendencia global (¿el mercado de fondo es alcista, bajista o lateral?)

### Hora de entrada como variable de análisis (no de ejecución todavía)

Se etiqueta cada operación simulada con su hora UTC y su sesión de mercado (Asia,
Londres, Nueva York, solapamiento Londres-NY, fuera de horario) y se reporta el
desglose de rendimiento por hora/sesión — tal como ya mostraba el reporte viejo
(`resources/report.md`, sección "Desglose por hora de entrada" y "por sesión"). Es
decir: **solo análisis por ahora**, con zona horaria de referencia **UTC**. Si algún
patrón horario resulta consistente y significativo, se evalúa más adelante si vale la
pena convertirlo en filtro de entrada.

### Componentes del motor de análisis técnico

1. **Estructura de mercado:** reglas objetivas de código (no "a ojo") para
   máximos/mínimos crecientes o decrecientes, rupturas de estructura, zonas de
   soporte/resistencia.
2. **Fibonacci:** usado como zona de confluencia (filtro de interés), nunca como
   señal aislada.
3. **Patrones de vela:** solo se consideran combinados con el contexto de estructura
   (en qué zona aparecen), nunca como señal por sí solos.
4. **Presión compradora/vendedora:** volumen por vela como proxy directo; si el
   volumen de Binance no alcanza, se complementa con funding rate (ver fuentes de
   datos abajo) como segundo proxy de sesgo comprador/vendedor.
5. Referencia bibliográfica ya disponible en `resources/12_claves_analisis_tecnico.pdf`
   (Dow, soporte/resistencia, figuras de precio, Elliott, osciladores, ADX, money
   management) — se usa como fuente de reglas concretas al implementar cada
   componente. (`resources/Mercado_analisi.pdf` es un apunte genérico de estudio de
   mercado/marketing, no aporta a esta parte técnica — se deja archivado sin uso.)

### Fuentes de datos adicionales (todas gratuitas, ya evaluadas)

- **Funding rate** (Binance Futures, vía `ccxt` — misma librería que ya se usaba
  para velas): se actualiza cada 8h, disponible como historial completo.
- **Fear & Greed Index** (`alternative.me`, API gratuita): un valor diario.
- *Descartado por ahora:* flujos on-chain y sentimiento social granular — requieren
  servicios pagos (Glassnode/CryptoQuant/Santiment) o un proyecto propio de etiquetado
  de wallets. Se deja para una eventual etapa posterior si el resto muestra señal.

### Herramientas estadísticas (upgrade sobre comparar grupos sueltos)

- **Autocorrelación:** ¿el retorno de hoy predice el de mañana?, a través de varios rezagos.
- **Regresión:** ajustar retorno futuro = f(retornos pasados, volumen, volatilidad) y
  medir R² real, en vez de un veredicto binario.
- **Causalidad de Granger:** ¿el funding rate o el Fear & Greed tienen poder
  predictivo formal sobre el precio, más allá de simple correlación?

Autocorrelación y regresión se pueden aplicar ya con los datos de velas que se van a
descargar; Granger se aplica una vez que funding rate/Fear & Greed estén integrados.

### Rigor de backtesting (para no repetir los problemas del intento anterior)

- Walk-forward / validación out-of-sample (no ajustar y testear sobre el mismo tramo).
- Costos de comisión y slippage reales de Binance incluidos en cada simulación.
- Tamaños de muestra suficientes antes de sacar conclusiones — el reporte anterior
  tenía escenarios de 18 y 7 operaciones, insuficiente para concluir nada; los nuevos
  escenarios deben apuntar a un mínimo razonable de operaciones (a definir con datos
  reales, pero claramente más que decenas).
- Reporte por escenario × mercado × timeframe × hora/sesión, con las mismas métricas
  que ya se usaban (win-rate, profit factor, expectancy, max drawdown, retorno total).

## Stack técnico

- **Lenguaje:** Python.
- **Datos:** Binance vía API pública (`ccxt`) para velas + funding rate;
  `alternative.me` para Fear & Greed. Sin credenciales necesarias en esta fase
  (solo lectura de datos públicos).
- Sin ejecución de órdenes en esta fase — Fase 1 es 100% lectura y simulación.

## Estructura de proyecto propuesta

```
illari/
├── resources/              # (ya existe) PDFs de referencia + reporte viejo
├── data/                   # velas/funding/F&G descargados, cacheados en disco
├── src/
│   ├── data/                # fetcher.py (Binance/ccxt), funding.py, fear_greed.py
│   ├── analysis/             # estructura.py, fibonacci.py, velas.py, volumen.py
│   ├── stats/                 # autocorrelacion.py, regresion.py, granger.py
│   ├── backtest/              # motor de simulación, walk-forward, métricas
│   └── report/                # generación de report.md + gráficos interactivos
├── config/                  # símbolos, escenarios de fecha, timeframes, sesiones UTC
├── tests/
└── requirements.txt
```

Esto es un punto de partida razonable, ajustable durante la implementación real.

## Próximos pasos inmediatos (una vez aprobado este plan)

1. Crear `requirements.txt` y estructura de carpetas (`src/`, `data/`, `config/`).
2. Implementar `src/data/fetcher.py` (velas OHLCV multi-timeframe vía `ccxt`/Binance)
   y validar fechas reales de listado de cada uno de los 5 símbolos.
3. Implementar `src/analysis/estructura.py` (soportes/resistencias, tendencia) como
   primer componente objetivo y testeable de forma aislada.
4. Backtest mínimo end-to-end sobre BTC/USDT en 1 timeframe, para validar el motor
   completo (datos → señal → simulación → reporte) antes de escalar a 5 mercados ×
   3 timeframes.
5. Recién ahí sumar Fibonacci, velas, volumen, funding rate, Fear & Greed, y las
   herramientas estadísticas.

## Verificación

- Cada módulo de `src/analysis` y `src/stats` con tests unitarios sobre datos
  sintéticos (casos donde se sabe de antemano cuál debe ser el resultado).
- El backtest mínimo (paso 4) debe correr de punta a punta y producir un
  `report.md` con métricas + desglose por hora/sesión, igual que el reporte viejo
  pero regenerado desde código versionado en este repo.
- Antes de dar por buena cualquier estrategia: revisar tamaño de muestra y
  performance out-of-sample, no solo el resultado in-sample.
