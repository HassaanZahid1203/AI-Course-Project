import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_confusion_matrix(cm, classes, title, ax):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=classes, yticklabels=classes)
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

def plot_classification_report(cr_text, title, ax):
    # Parse the classification report text
    lines = cr_text.split('\n')
    classes = []
    metrics = []
    
    for line in lines[2:-3]:  # Skip header and footer lines
        if line.strip():
            parts = line.split()
            # Handle lines with class names that might contain spaces
            if len(parts) >= 5:
                # The class name might be multiple words (like "East Asian")
                # Find where the numbers start
                for i in range(1, len(parts)):
                    try:
                        # Try to convert to float to find where numbers start
                        float(parts[i])
                        class_name = ' '.join(parts[:i])
                        metrics_values = [float(x) for x in parts[i:i+3]]
                        classes.append(class_name)
                        metrics.append(metrics_values)
                        break
                    except ValueError:
                        continue
    
    metrics = np.array(metrics)
    im = ax.imshow(metrics, interpolation='nearest', cmap='YlGnBu', vmin=0, vmax=1)
    ax.set_title(title)
    
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(['Precision', 'Recall', 'F1-score'])
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)
    
    plt.colorbar(im, ax=ax)

def visualize_metrics(gender_cm, gender_cr, race_cm, race_cr, age_cm, age_cr, dataset_name):
    fig = plt.figure(figsize=(20, 25))
    fig.suptitle(f'Model Evaluation Metrics - {dataset_name}', fontsize=16, y=1.02)
    
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1, 1.5], height_ratios=[1, 1, 1])
    
    # Gender
    ax1 = fig.add_subplot(gs[0, 0])
    plot_confusion_matrix(gender_cm, ['Male', 'Female'], 'Gender Confusion Matrix', ax1)
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_classification_report(gender_cr, 'Gender Classification Report', ax2)
    
    # Race
    race_classes = ['Black', 'East Asian', 'Indian', 'Latino_Hispanic', 
                   'Middle Eastern', 'Southeast Asian', 'White']
    ax3 = fig.add_subplot(gs[1, 0])
    plot_confusion_matrix(race_cm, race_classes, 'Race Confusion Matrix', ax3)
    
    ax4 = fig.add_subplot(gs[1, 1])
    plot_classification_report(race_cr, 'Race Classification Report', ax4)
    
    # Age
    age_classes = ['0-2', '10-19', '20-29', '3-9', '30-39', '40-49', '50-59', '60-69', 'more than 70']
    ax5 = fig.add_subplot(gs[2, 0])
    plot_confusion_matrix(age_cm, age_classes, 'Age Confusion Matrix (±1)', ax5)
    
    ax6 = fig.add_subplot(gs[2, 1])
    plot_classification_report(age_cr, 'Age Classification Report (±1)', ax6)
    
    plt.tight_layout()
    plt.savefig(f'{dataset_name.lower().replace(" ", "_")}_evaluation.png', bbox_inches='tight', dpi=300)
    plt.close()

# Training Dataset Metrics
train_gender_cm = np.array([[45268, 718], [1763, 38995]])
train_gender_cr = """
              precision    recall  f1-score   support

        Male       0.96      0.98      0.97     45986
      Female       0.98      0.96      0.97     40758

    accuracy                           0.97     86744
   macro avg       0.97      0.97      0.97     86744
weighted avg       0.97      0.97      0.97     86744
"""

train_race_cm = np.array([
    [11675, 9, 187, 255, 55, 18, 34],
    [36, 10507, 11, 533, 56, 980, 164],
    [432, 13, 10020, 1286, 449, 60, 59],
    [467, 68, 458, 10800, 756, 121, 697],
    [31, 5, 82, 802, 7528, 5, 763],
    [297, 1902, 146, 1305, 63, 7039, 43],
    [52, 60, 56, 1576, 1087, 15, 13681]
])

train_race_cr = """
                 precision    recall  f1-score   support

          Black       0.90      0.95      0.93     12233
     East Asian       0.84      0.86      0.85     12287
         Indian       0.91      0.81      0.86     12319
Latino_Hispanic       0.65      0.81      0.72     13367
 Middle Eastern       0.75      0.82      0.78      9216
Southeast Asian       0.85      0.65      0.74     10795
          White       0.89      0.83      0.86     16527

       accuracy                           0.82     86744
      macro avg       0.83      0.82      0.82     86744
   weighted avg       0.83      0.82      0.82     86744
"""

train_age_cm = np.array([
    [1288, 0, 2, 501, 0, 0, 0, 1, 0],
    [0, 7454, 0, 1577, 48, 14, 9, 1, 0],
    [0, 0, 23537, 0, 1954, 98, 8, 1, 0],
    [57, 340, 0, 10005, 0, 2, 3, 1, 0],
    [0, 286, 7158, 0, 11727, 0, 71, 8, 0],
    [1, 43, 527, 19, 0, 10115, 0, 39, 0],
    [0, 5, 31, 8, 184, 0, 5992, 0, 8],
    [0, 1, 4, 2, 11, 116, 0, 2645, 0],
    [0, 0, 0, 0, 0, 2, 36, 0, 804]
])

train_age_cr = """
              precision    recall  f1-score   support

         0-2       0.96      0.72      0.82      1792
       10-19       0.92      0.82      0.87      9103
       20-29       0.75      0.92      0.83     25598
         3-9       0.83      0.96      0.89     10408
       30-39       0.84      0.61      0.71     19250
       40-49       0.98      0.94      0.96     10744
       50-59       0.98      0.96      0.97      6228
       60-69       0.98      0.95      0.97      2779
more than 70       0.99      0.95      0.97       842

    accuracy                           0.85     86744
   macro avg       0.91      0.87      0.89     86744
weighted avg       0.86      0.85      0.84     86744
"""

# Validation Dataset Metrics
val_gender_cm = np.array([[5450, 342], [540, 4622]])
val_gender_cr = """
              precision    recall  f1-score   support

        Male       0.91      0.94      0.93      5792
      Female       0.93      0.90      0.91      5162

    accuracy                           0.92     10954
   macro avg       0.92      0.92      0.92     10954
weighted avg       0.92      0.92      0.92     10954
"""

val_race_cm = np.array([
    [1344, 8, 46, 97, 27, 14, 20],
    [20, 1139, 8, 112, 26, 196, 49],
    [121, 10, 957, 272, 102, 26, 28],
    [86, 23, 116, 996, 162, 40, 200],
    [13, 6, 30, 261, 658, 4, 237],
    [75, 323, 48, 242, 15, 682, 30],
    [21, 32, 25, 344, 267, 8, 1388]
])

val_race_cr = """
                 precision    recall  f1-score   support

          Black       0.80      0.86      0.83      1556
     East Asian       0.74      0.73      0.74      1550
         Indian       0.78      0.63      0.70      1516
Latino_Hispanic       0.43      0.61      0.50      1623
 Middle Eastern       0.52      0.54      0.53      1209
Southeast Asian       0.70      0.48      0.57      1415
          White       0.71      0.67      0.69      2085

       accuracy                           0.65     10954
      macro avg       0.67      0.65      0.65     10954
   weighted avg       0.67      0.65      0.66     10954
"""

val_age_cm = np.array([
    [116, 0, 1, 79, 0, 1, 1, 1, 0],
    [0, 883, 0, 266, 27, 3, 0, 2, 0],
    [0, 0, 2752, 0, 474, 68, 4, 2, 0],
    [26, 125, 0, 1199, 0, 4, 1, 1, 0],
    [0, 87, 967, 0, 1234, 0, 36, 6, 0],
    [0, 13, 173, 8, 0, 1143, 0, 15, 1],
    [0, 3, 24, 3, 77, 0, 682, 0, 7],
    [0, 1, 3, 1, 6, 44, 0, 266, 0],
    [0, 0, 0, 0, 0, 5, 15, 0, 98]
])

val_age_cr = """
              precision    recall  f1-score   support

         0-2       0.82      0.58      0.68       199
       10-19       0.79      0.75      0.77      1181
       20-29       0.70      0.83      0.76      3300
         3-9       0.77      0.88      0.82      1356
       30-39       0.68      0.53      0.59      2330
       40-49       0.90      0.84      0.87      1353
       50-59       0.92      0.86      0.89       796
       60-69       0.91      0.83      0.87       321
more than 70       0.92      0.83      0.88       118

    accuracy                           0.76     10954
   macro avg       0.82      0.77      0.79     10954
weighted avg       0.77      0.76      0.76     10954
"""

# Visualize both datasets
visualize_metrics(train_gender_cm, train_gender_cr, train_race_cm, train_race_cr, train_age_cm, train_age_cr, "Training Dataset")
visualize_metrics(val_gender_cm, val_gender_cr, val_race_cm, val_race_cr, val_age_cm, val_age_cr, "Validation Dataset")