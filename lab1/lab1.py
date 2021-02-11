import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Solution:
    def __init__(self) -> None:
        # TODO: DONE
        # Load data from data/chipotle.tsv file using Pandas library and 
        # assign the dataset to the 'chipo' variable.
        file = 'data/chipotle.tsv'
        self.chipo = pd.read_csv(file, sep='\t')
    
    def top_x(self, count) -> None:
        # TODO: DONE
        # Top x number of entries from the dataset and display as markdown format.
        topx = self.chipo.head(count)
        print(topx.to_markdown())
        
    def count(self) -> int:
        # TODO: DONE
        # The number of observations/entries in the dataset.
        return self.chipo.shape[0]
        # return -1
    
    def info(self) -> None:
        # TODO: DONE
        # print data info.
        print(pd.DataFrame.info(self.chipo))
        # pass
    
    def num_column(self) -> int:
        # TODO: DONE
        # return the number of columns in the dataset
        return self.chipo.shape[1]
        # return -1
    
    def print_columns(self) -> None:
        # TODO: DONE
        # Print the name of all the columns.
        print(list(self.chipo.columns))
        # pass
    
    def most_ordered_item(self):
        # TODO: QUESTION REMAIN
        item_name = None
        order_id = -1
        quantity = -1
        # item_q = self.chipo.groupby(['item_name']).agg({'quantity': 'sum'})
        # item_name = item_q.sort_values('quantity',ascending=False)[:1].index.values[0]
        item_q = self.chipo.groupby('item_name').sum().sort_values('quantity',ascending=False).head(1)
        # item_name = list(self.chipo.item_name.mode())[0]
        item_name = item_q.index.values[0]
        order_id = item_q.values[0][0]
        quantity = item_q.values[0][1]
        return item_name, order_id, quantity

    def total_item_orders(self) -> int:
       # TODO: DONE
       # How many items were orderd in total?
       return self.chipo.quantity.sum()
   
    def total_sales(self) -> float:
        # TODO: Done
        # 1. Create a lambda function to change all item prices to float.
        # 2. Calculate total sales.
        self.chipo.item_price = self.chipo.item_price.apply(lambda row: float(row.lstrip('$')))
        return (self.chipo.item_price * self.chipo.quantity).sum()
   
    def num_orders(self) -> int:
        # TODO: DONE
        # How many orders were made in the dataset?
        return len(list(self.chipo.order_id.value_counts()))
    
    def average_sales_amount_per_order(self) -> float:
        # TODO: DONE
        num_orders = self.num_orders()
        total_sales = (self.chipo.item_price * self.chipo.quantity).sum()
        return (total_sales / num_orders).round(2)

    def num_different_items_sold(self) -> int:
        # TODO: DONE
        # How many different items are sold?
        return self.chipo.item_name.nunique()
    
    def plot_histogram_top_x_popular_items(self, x:int) -> None:
        from collections import Counter
        letter_counter = Counter(self.chipo.item_name)
        # TODO: DONE
        # 1. convert the dictionary to a DataFrame
        # 2. sort the values from the top to the least value and slice the first 5 items
        # 3. create a 'bar' plot from the DataFrame
        # 4. set the title and labels:
        #     x: Items
        #     y: Number of Orders
        #     title: Most popular items
        # 5. show the plot. Hint: plt.show(block=True).
        df = pd.DataFrame.from_dict(letter_counter, orient='index').reset_index()
        top_five = df.nlargest(x, [0])
        print(top_five)
        ax = top_five.plot.bar(x='index', rot=0)
        plt.title("Most popular items")
        plt.xlabel("Items")
        plt.ylabel("Number of Orders")
        plt.show(block=True)
        pass
        
    def scatter_plot_num_items_per_order_price(self) -> None:
        # TODO: DONE
        # 1. create a list of prices by removing dollar sign and trailing space.
        # 2. groupby the orders and sum it.
        # 3. create a scatter plot:
        #       x: orders' item price
        #       y: orders' quantity
        #       s: 50
        #       c: blue
        # 4. set the title and labels.
        #       title: Numer of items per order price
        #       x: Order Price
        #       y: Num Items
        df = self.chipo.groupby(['order_id']).sum()
        ax1 = df.plot.scatter(x='item_price', y='quantity', s=50, c='blue')
        plt.title("Numer of items per order price")
        plt.xlabel("Order Price")
        plt.ylabel("Num Items")
        plt.show(block=True)
        pass
    
        

def test() -> None:
    solution = Solution()
    solution.top_x(10)
    count = solution.count()
    print(count)
    assert count == 4622
    solution.info()
    count = solution.num_column()
    print(count)
    assert count == 5
    solution.print_columns()
    item_name, order_id, quantity = solution.most_ordered_item()
    assert item_name == 'Chicken Bowl'
    assert order_id == 713926	
    # assert quantity == 159
    total = solution.total_item_orders()
    assert total == 4972
    assert 39237.02 == solution.total_sales()
    assert 1834 == solution.num_orders()
    assert 21.39 == solution.average_sales_amount_per_order()
    assert 50 == solution.num_different_items_sold()
    solution.plot_histogram_top_x_popular_items(5)
    solution.scatter_plot_num_items_per_order_price()

    
if __name__ == "__main__":
    # execute only if run as a script
    test()
    
    