class EDA:
    def __init__(self, dataset):
        self.dataset = dataset

    def summary_statistics(self):
        print("Summary Statistics:")
        print(self.dataset.describe(include='all'))

    def missing_values(self):
        print("Missing Values:")
        print(self.dataset.isnull().sum())

    def visualize_distributions(self):
        for column in self.dataset.select_dtypes(include=[np.number]).columns:
            plt.figure(figsize=(10, 5))
            sns.histplot(self.dataset[column], bins=30, kde=True)
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()

    def visualize_boxplots(self):
        for column in self.dataset.select_dtypes(include=[np.number]).columns:
            plt.figure(figsize=(10, 5))
            sns.boxplot(x=self.dataset[column])
            plt.title(f'Boxplot of {column}')
            plt.xlabel(column)
            plt.show()

    def correlation_heatmap(self):
        plt.figure(figsize=(12, 8))
        correlation_matrix = self.dataset.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title('Correlation Heatmap')
        plt.show()
