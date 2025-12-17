# Manual profesional de IA — Términos, conceptos y buenas prácticas

**Objetivo:** Documento de referencia exhaustiva (para profesionales) con definiciones técnicas, explicaciones simples con analogías, fórmulas clave, fragmentos de código o pseudocódigo, buenas prácticas, trade‑offs y checklists operativos.

---

## Índice

1. Fundamentos y notación
2. Arquitecturas y componentes
3. Optimización y entrenamiento
4. Regularización, normalización y estabilización
5. Representaciones y aprendizaje no supervisado
6. Modelos generativos
7. Evaluación, métricas y calibración
8. Incertidumbre y robustez
9. Seguridad, adversarialidad y certificación
10. Privacidad, federación y compliance
11. Compresión y despliegue (edge / producción)
12. Tokenización y NLP moderno
13. Infraestructura, MLOps y monitoreo
14. Sesgos, ética y gobernanza
15. Teoría avanzada y fenómenos emergentes
16. Checklists operativos y pruebas
17. Fragmentos de código y pseudocódigo útiles
18. Glosario extendido

---

# 1. Fundamentos y notación

### 1.1. Parámetros (pesos) y sesgos

* **Definición técnica:** Valores numéricos {
  $w_{i}$, $b$ } que parametrizan funciones del modelo. Se actualizan por optimización para minimizar una pérdida.
* **Analogía:** Cantidades en una receta; cambias cantidades (pesos) para ajustar el sabor (salida).

### 1.2. Función de activación

* **Definición:** No linealidad aplicada a la salida de una capa: ReLU, Sigmoid, Tanh, GELU, SiLU/Swish.
* **Por qué importa:** Introducen no linealidad; afectan gradiente, saturación y convergencia.

### 1.3. Pérdida (loss)

* **Tipos comunes:** MSE, Cross‑Entropy, Hinge, Contrastive/InfoNCE, Triplet loss.
* **Nota:** Elegir loss que refleje la métrica final que importa (ej. ranking, F1, AUC).

### 1.4. Backpropagation

* **Resumen:** Aplicación de la regla de la cadena para computar gradientes y actualizar parámetros.
* **Analogía:** Retroalimentación en un proceso de ensayo y error.

---

# 2. Arquitecturas y componentes

### 2.1. MLP (Perceptrón multicapa)

* **Uso:** Baseline para tabular y problemas sencillos.
* **Limitación:** No captura estructura espacial/temporal eficientemente.

### 2.2. CNN (Redes convolucionales)

* **Fundamento:** Convoluciones para explotar invariancias locales (
  peso compartido). Pooling y convoluciones.
* **Uso:** Visión por computador, audio (espectrogramas).

### 2.3. RNN / LSTM / GRU

* **Por qué:** Manejo de secuencias; LSTM/GRU corrigen problema de gradiente en RNN simple.
* **Limitación:** Escala mal a largas dependencias; reemplazados en muchas tareas por transformers.

### 2.4. Transformer y Self‑Attention

* **Componentes:** Multi‑head attention, feed‑forward, residuals, LayerNorm.
* **Por qué revolucionó:** Captura relaciones a distancia de forma paralelizable.
* **Variantes:** Encoder (BERT), Decoder (GPT), Encoder‑Decoder (T5), Vision Transformer (ViT).

### 2.5. Mecanismos extendidos de atención

* **Sparse attention:** reduce costo en long context. **Relative positional encodings** para invariancia de distancia.
* **Linformer, Performer, Longformer:** aproximaciones para tokens largos.

---

# 3. Optimización y entrenamiento

### 3.1. Optimizadores

* **SGD (con momentum):** Simple y robusto; ruido útil para generalización.
* **Adam / AdamW / Adafactor / RMSprop:** Adaptativos; AdamW corrige regularización (weight decay separado).
* **Buena práctica:** Usar AdamW con weight decay y scheduler (warmup + decay).

### 3.2. Scheduler de tasa de aprendizaje

* **Warmup:** evita oscilaciones tempranas; crucial en transformers.
* **Cosine decay, linear decay, cyclical LR.**

### 3.3. Gradient clipping

* **Por qué:** Evitar explosión de gradientes (RNNs, training grande).
* **Cómo:** Clip by value o by norm.

### 3.4. Batch size y efectos

* Batch grande reduce ruido del gradiente y requiere ajustar LR (regla lineal / LARS).
* **Gradient noise scale** guía elección de batch size.

### 3.5. Checkpointing y recuperación

* Guardar pesos, optimizador, scheduler y RNG seeds para reproducibilidad.

---

# 4. Regularización, normalización y estabilización

### 4.1. Dropout

* Random deactivations para reducir co‑adaptación.

### 4.2. BatchNorm, LayerNorm, GroupNorm

* **BatchNorm:** mejor para CNNs con batch estable. **LayerNorm:** usado en Transformers.
* **Trade‑off:** BatchNorm depende de batch size; LayerNorm es determinista por ejemplo.

### 4.3. Weight decay (L2) y AdamW

* Penaliza magnitud de pesos; AdamW separa penalización del paso adaptativo para corregir sesgos.

### 4.4. Label smoothing

* Reduce confianza extrema, mejora calibración.

### 4.5. Early stopping y regularización temprana

* Monitoreo en validación para cortar entrenamiento cuando la generalización empeora.

---

# 5. Representaciones y aprendizaje no supervisado

### 5.1. Autoencoders y Variational Autoencoders (VAE)

* **Autoencoder:** compresión y reconstrucción.
* **VAE:** añade estructura probabilística; optimiza ELBO = reconstrucción - KL(q||p).

### 5.2. Contrastive Learning (SimCLR, MoCo, CLIP)

* **Idea:** Aprender embeddings que acerquen positivos y separen negativos. InfoNCE es la pérdida típica.
* **Uso:** Representaciones transferibles sin etiquetas.

### 5.3. Self‑supervised objectives

* Masked LM (BERT), next token (GPT), image patch prediction (MAE), rotation prediction, jigsaw.

---

# 6. Modelos generativos

### 6.1. GANs (Generative Adversarial Networks)

* **Estructura:** generador vs discriminador en juego min‑max.
* **Problemas:** inestabilidad, modo colapso.

### 6.2. Flow‑based models

* Transformaciones invertibles con probabilidades explícitas (RealNVP, Glow).

### 6.3. Autoregresivos

* Modelos que factoran p(x) = \prod p(x_i | x_{<i}). GPT es ejemplo.

### 6.4. Diffusion models

* Proceso de ruido y denoise (SDE/score matching). Lideran generación de imágenes recientes.

---

# 7. Evaluación, métricas y calibración

### 7.1. Métricas básicas

* Accuracy, Precision, Recall, F1, Confusion Matrix, ROC/AUC.
* Para ranking: MRR, MAP, NDCG.

### 7.2. NLP: BLEU, ROUGE, METEOR

* Para generación; entender sus limitaciones (no capturan factualidad ni coherencia).

### 7.3. Perplexity (LM)

* Exponent of cross‑entropy; mide sorpresa.

### 7.4. Calibración (ECE, Brier)

* Medir si probabilidades correspondan a frecuencias reales.
* Técnicas: temperature scaling, isotonic regression.

---

# 8. Incertidumbre y robustez

### 8.1. Aleatoric vs Epistemic

* Aleatoric: ruido inherente. Epistemic: falta de conocimiento (reducible con datos).

### 8.2. Técnicas para estimar incertidumbre

* **MC Dropout:** usar dropout en inferencia para muestras.
* **Deep Ensembles:** múltiples modelos para estimar varianza.
* **Bayesian neural networks:** aproximación de posterior.

### 8.3. Out‑of‑Distribution detection

* Softmax escalar no es suficiente; usar métodos basados en embeddings, ODIN, energy‑based scores.

---

# 9. Seguridad, adversarialidad y certificación

### 9.1. Adversarial attacks

* **FGSM:** Fast Gradient Sign Method; **PGD:** Projected Gradient Descent.
* **White‑box vs Black‑box.**

### 9.2. Adversarial training

* Incluir adversarial examples en el entrenamiento; costoso pero efectivo.

### 9.3. Certified defenses

* Randomized smoothing y técnicas con garantías formales para perturbaciones L2/L∞.

---

# 10. Privacidad, federación y compliance

### 10.1. Differential Privacy (DP)

* **Idea:** añadir ruido a gradientes o agregados para que la presencia de un ejemplo no sea distinguible.
* **DP‑SGD:** clip grads y añadir ruido calibrado.

### 10.2. Federated Learning

* Entrenamiento distribuido en clientes que no comparten datos crudos; requiere orquestación (agg segura, compresión, defensa contra poisoning).

### 10.3. Compliance y regulaciones

* GDPR, LGPD, y regulaciones sectoriales exigen trazabilidad y minimización de datos.

---

# 11. Compresión y despliegue (edge / producción)

### 11.1. Pruning

* Eliminar conexiones/pesos insignificantes. Estrategias: magnitude pruning, structured pruning.

### 11.2. Quantization

* Reducir precisión (float32 → float16 → int8). Trade‑off: velocidad y tamaño vs precisión.

### 11.3. Distillation

* Entrenar student para imitar teacher; útil para reducción de tamaño y latencia.

### 11.4. Pipeline de inferencia

* Batching, caching, model partitioning (tensor parallel, pipeline parallel), offloading a CPU/NPU.

### 11.5. Observabilidad: latency, throughput, P99/P50, memoria, consumo energético

* Establecer SLOs y alertas.

---

# 12. Tokenización y NLP moderno

### 12.1. Métodos de tokenización

* **BPE (Byte Pair Encoding), WordPiece, Unigram, byte‑level.**
* **Efectos:** longitud de secuencia, OOV handling, subword fragmentation.

### 12.2. Embeddings

* Representación vectorial de tokens/órdenes. Técnicas: static (Word2Vec, GloVe) vs contextual (BERT, GPT).

### 12.3. Retrieval‑Augmented Generation (RAG)

* Combina búsqueda en vectores + LM para respuestas con grounding en documentos.
* Componentes: index vectorial (FAISS/HNSW), retriever (dense/sparse), ranker.

---

# 13. Infraestructura, MLOps y monitoreo

### 13.1. Versionado y reproducibilidad

* Git para código, DVC/MLflow para datos/modelos, registries para artefactos.

### 13.2. CI/CD para ML

* Tests unitarios, tests de datos, validaciones de modelos, despliegue canario, rollback.

### 13.3. Monitoreo en producción

* Métricas online, alertas de drift, validación de input schema, logs de predicción.

### 13.4. Orquestación y recursos

* Contenedores (Docker/Podman), orquestadores (K8s o alternativas ligeras), infra serverless para inferencia.

---

# 14. Sesgos, ética y gobernanza

### 14.1. Tipos de sesgo

* Data bias (recolección), label bias, measurement bias.

### 14.2. Mitigación

* Recolección diversa, re‑pesos, fairness regularizers, auditing y tests A/B controlados.

### 14.3. Documentación

* Datasheets for datasets, Model cards, Risk assessment y registro de decisiones.

---

# 15. Teoría avanzada y fenómenos emergentes

### 15.1. Double Descent

* Fenómeno donde aumentar capacidad primero empeora y luego mejora generalización.

### 15.2. Lottery Ticket Hypothesis

* Existen sub‑redes inicializadas que alcanzan performance comparable tras reentrenamiento.

### 15.3. Scaling laws

* Leyes empíricas sobre cómo performance escala con datos, parámetros y compute.

### 15.4. Information Bottleneck

* Balance entre compresión de la representación y retención de información relevante.

---

# 16. Checklists operativos y pruebas

### 16.1. Checklist de entrenamiento (pre‑run)

* Dataset: calidad, balance, split (train/val/test), seed.
* Augmentations: definidas y guardables.
* Hyperparams: LR, batch size, weight decay, scheduler.
* Checkpointing: frecuencia y retención.
* Tests rápidos: entrenamiento por 1 epoch con subset para detectar errores.

### 16.2. Checklist de validación antes de push a producción

* Pruebas de regresión en métricas clave.
* Calibración y ECE.
* Test OOD y adversarial básico.
* Test de latencia y memoria.
* Documentación y model card.

### 16.3. Checklist de monitoreo post‑despliegue

* Drift detection (input feature distribution).
* Alertas si cambio en A/B test o drop en accuracy.
* Logs de errores y tasa de abstención.

---

# 17. Fragmentos de código y pseudocódigo útiles

### 17.1. AdamW (PyTorch‑like pseudocode)

```python
# pseudocódigo conceptual
for batch in data:
    loss = model(batch)
    loss.backward()
    for p in model.params:
        # moment estimates
        m[p] = beta1 * m[p] + (1-beta1) * grad[p]
        v[p] = beta2 * v[p] + (1-beta2) * (grad[p]**2)
        m_hat = m[p] / (1 - beta1**t)
        v_hat = v[p] / (1 - beta2**t)
        # update with weight decay separado
        p -= lr * (m_hat / (sqrt(v_hat)+eps) + weight_decay * p)
```

### 17.2. Mixup

```python
# x1,y1 and x2,y2 random pairs
lam = Beta(alpha, alpha)
x_new = lam * x1 + (1-lam) * x2
y_new = lam * y1 + (1-lam) * y2
```

### 17.3. InfoNCE (contrastive loss)

```python
# z_i: embeddings, positives in same batch
logits = sim(z_i, z_pos)/tau
labels = 0  # positive index
loss = CrossEntropy(logits, labels)
```

### 17.4. DP‑SGD (esquema)

```text
for each microbatch:
  grads = compute_gradients()
  grads = clip_by_norm(grads, C)
  aggregated = sum(grads)
  noisy = aggregated + Normal(0, sigma*C)
  update = noisy / batch_size
  optimizer.step(update)
```

### 17.5. Pruning (magnitude)

```python
# Compute absolute magnitudes, zero lowest k%
magnitudes = abs(model.weights)
threshold = percentile(magnitudes, k)
model.weights[magnitudes < threshold] = 0
```

---

# 18. Glosario extendido (selección rápida)

* **AUC:** Area under ROC curve.
* **Backbone:** arquitectura base (ResNet, ViT).
* **Catastrophic forgetting:** olvidar tareas pasadas al aprender nuevas.
* **Checkpoint:** snapshot del estado de entrenamiento.
* **Data leakage:** filtración de información del set de test al entrenamiento.
* **Embedding:** vector numérico que representa una unidad semántica.
* **Feature store:** almacenamiento centralizado para features computadas.
* **Gradient clipping:** limitar norma del gradiente.
* **Head:** capa final del modelo para una tarea.
* **IoU:** Intersection over Union (visión).
* **OOD:** Out‑of‑distribution.
* **RAG:** Retrieval augmented generation.
* **SOTA:** State of the art.

---

# Conclusión y pasos siguientes

Este manual es una base técnica y operativa. Puedo:

* Generar un **notebook reproducible** con ejemplos ejecutables (Mixup, AdamW vs Adam, InfoNCE, DP‑SGD, pruning + cuantización).
* Convertir el manual en **presentación** (PPTX) o en una **cheat‑sheet** de 1 página.
* Diseñar tests automáticos y scripts (CI) para tus pipelines.

Dime cuál de las opciones preferís y lo entrego listo para usar.

---

*Fin del documento.*
