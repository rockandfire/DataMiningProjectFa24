import tkinter as tk
from tkinter import ttk, scrolledtext
from tkinter import messagebox
import pandas as pd
from typing import Dict, List
import threading
from testing import MTGDeckBuilder


class MTGDeckBuilderGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MTG Deck Builder Assistant")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')

        # Initialize deck builder
        self.deck_builder = MTGDeckBuilder(debug=False)

        # Create main frames
        self.create_frames()
        self.create_commander_section()
        self.create_results_section()
        self.create_future_card_section()

        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_frames(self):
        """Create main layout frames"""
        # Left panel for commander selection
        self.left_frame = ttk.Frame(self.root, padding="10")
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Center panel for results
        self.center_frame = ttk.Frame(self.root, padding="10")
        self.center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right panel for future card analysis
        self.right_frame = ttk.Frame(self.root, padding="10")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y)

    def create_commander_section(self):
        """Create commander selection section"""
        # Commander Selection Frame
        commander_frame = ttk.LabelFrame(self.left_frame, text="Commander Selection", padding="10")
        commander_frame.pack(fill=tk.X, pady=5)

        # Commander search entry
        ttk.Label(commander_frame, text="Search Commander:").pack(fill=tk.X)
        self.commander_entry = ttk.Entry(commander_frame)
        self.commander_entry.pack(fill=tk.X, pady=5)

        # Commander list
        self.commander_listbox = tk.Listbox(commander_frame, height=15)
        self.commander_listbox.pack(fill=tk.BOTH, expand=True, pady=5)

        # Analyze button
        self.analyze_btn = ttk.Button(commander_frame, text="Analyze Commander",
                                      command=self.analyze_commander)
        self.analyze_btn.pack(fill=tk.X, pady=5)

        # Bind search functionality
        self.commander_entry.bind('<KeyRelease>', self.search_commander)

    def create_results_section(self):
        """Create results display section with enhanced card information"""
        # Results notebook
        self.results_notebook = ttk.Notebook(self.center_frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)

        # Recommended Cards tab
        self.recommendations_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.recommendations_frame, text="Recommended Cards")

        # Create treeview for recommendations with more columns
        self.recommendations_tree = ttk.Treeview(
            self.recommendations_frame,
            columns=("Name", "Rank", "Score", "MV", "Text", "Keywords"),
            show="headings",
            selectmode="browse"
        )

        # Configure columns
        self.recommendations_tree.heading("Name", text="Card Name")
        self.recommendations_tree.heading("Rank", text="EDHRec Rank")
        self.recommendations_tree.heading("Score", text="Similarity Score")
        self.recommendations_tree.heading("MV", text="Mana Value")
        self.recommendations_tree.heading("Text", text="Card Text")
        self.recommendations_tree.heading("Keywords", text="Keywords")

        # Set column widths
        self.recommendations_tree.column("Name", width=150)
        self.recommendations_tree.column("Rank", width=100)
        self.recommendations_tree.column("Score", width=100)
        self.recommendations_tree.column("MV", width=80)
        self.recommendations_tree.column("Text", width=300)
        self.recommendations_tree.column("Keywords", width=150)

        # Add scrollbars
        y_scroll = ttk.Scrollbar(self.recommendations_frame, orient="vertical",
                                 command=self.recommendations_tree.yview)
        x_scroll = ttk.Scrollbar(self.recommendations_frame, orient="horizontal",
                                 command=self.recommendations_tree.xview)
        self.recommendations_tree.configure(yscrollcommand=y_scroll.set,
                                            xscrollcommand=x_scroll.set)

        # Pack scrollbars and tree
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.recommendations_tree.pack(fill=tk.BOTH, expand=True)

        # Card details frame below the tree
        self.card_details_frame = ttk.LabelFrame(self.recommendations_frame,
                                                 text="Card Details", padding="10")
        self.card_details_frame.pack(fill=tk.X, pady=5)

        # Text widget for detailed card information
        self.card_details_text = scrolledtext.ScrolledText(self.card_details_frame,
                                                           height=8, wrap=tk.WORD)
        self.card_details_text.pack(fill=tk.BOTH, expand=True)

        # Bind selection event
        self.recommendations_tree.bind('<<TreeviewSelect>>', self.show_card_details)

        # Clusters tab remains the same
        self.clusters_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.clusters_frame, text="Card Clusters")

        # Create treeview for clusters
        self.clusters_tree = ttk.Treeview(self.clusters_frame)
        self.clusters_tree.pack(fill=tk.BOTH, expand=True)

    def show_card_details(self, event):
        """Display detailed information about the selected card"""
        selected_items = self.recommendations_tree.selection()
        if not selected_items:
            return

        # Get the selected item's values
        item = self.recommendations_tree.item(selected_items[0])
        values = item['values']

        # Clear previous details
        self.card_details_text.delete('1.0', tk.END)

        # Display detailed card information
        details = f"""Card Name: {values[0]}
    EDHRec Rank: {values[1]}
    Similarity Score: {values[2]}
    Mana Value: {values[3]}

    Card Text:
    {values[4]}

    Keywords:
    {values[5]}

    """
        if len(values) > 6:  # If we have similarity details stored
            similarity_details = values[6]
            details += f"\nShared Mechanics:\n{similarity_details}"

        self.card_details_text.insert(tk.END, details)

    def create_future_card_section(self):
        """Create future card analysis section"""
        future_frame = ttk.LabelFrame(self.right_frame, text="Future Card Analysis", padding="10")
        future_frame.pack(fill=tk.X, pady=5)

        # Card name entry
        ttk.Label(future_frame, text="Card Name:").pack(fill=tk.X)
        self.future_name_entry = ttk.Entry(future_frame)
        self.future_name_entry.pack(fill=tk.X, pady=5)

        # Mana value entry
        ttk.Label(future_frame, text="Mana Value:").pack(fill=tk.X)
        self.mana_value_entry = ttk.Entry(future_frame)
        self.mana_value_entry.pack(fill=tk.X, pady=5)

        # Color identity selection
        ttk.Label(future_frame, text="Color Identity:").pack(fill=tk.X)
        colors_frame = ttk.Frame(future_frame)
        colors_frame.pack(fill=tk.X, pady=5)

        self.color_vars = {}
        for color in ['W', 'U', 'B', 'R', 'G']:
            self.color_vars[color] = tk.BooleanVar()
            ttk.Checkbutton(colors_frame, text=color,
                            variable=self.color_vars[color]).pack(side=tk.LEFT)

        # Card text entry
        ttk.Label(future_frame, text="Card Text:").pack(fill=tk.X)
        self.card_text_entry = scrolledtext.ScrolledText(future_frame, height=10)
        self.card_text_entry.pack(fill=tk.X, pady=5)

        # Analyze button
        self.analyze_future_btn = ttk.Button(future_frame, text="Analyze Future Card",
                                             command=self.analyze_future_card)
        self.analyze_future_btn.pack(fill=tk.X, pady=5)

        # Results display
        self.future_results = scrolledtext.ScrolledText(future_frame, height=10)
        self.future_results.pack(fill=tk.BOTH, expand=True, pady=5)

    def search_commander(self, event):
        """Search for commanders as user types"""
        search_term = self.commander_entry.get().lower()
        self.commander_listbox.delete(0, tk.END)

        commanders = self.deck_builder.get_valid_commanders()
        for commander in commanders[commanders.str.lower().str.contains(search_term)]:
            self.commander_listbox.insert(tk.END, commander)

    def analyze_commander(self):
        """Analyze selected commander"""
        if not self.commander_listbox.curselection():
            messagebox.showwarning("Selection Required", "Please select a commander to analyze.")
            return

        commander = self.commander_listbox.get(self.commander_listbox.curselection())
        self.status_var.set(f"Analyzing {commander}...")

        # Run analysis in separate thread
        def run_analysis():
            try:
                results = self.deck_builder.analyze_deck(commander)
                self.root.after(0, lambda: self.display_results(results))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.root.after(0, lambda: self.status_var.set("Ready"))

        threading.Thread(target=run_analysis, daemon=True).start()

    def display_results(self, results: Dict):
        """Display analysis results with enhanced card information"""
        # Clear previous results
        self.recommendations_tree.delete(*self.recommendations_tree.get_children())
        self.clusters_tree.delete(*self.clusters_tree.get_children())

        # Display recommended cards with full information
        for card in results['recommended_cards'][:20]:  # Top 20 recommendations
            # Get similarity details if available
            similarity_details = ""
            if 'similarity_details' in card:
                details = card['similarity_details']
                shared_mechanics = []
                if details['shared_counters']:
                    shared_mechanics.append(f"Counters: {', '.join(details['shared_counters'])}")
                if details['shared_triggers']:
                    shared_mechanics.append(f"Triggers: {', '.join(details['shared_triggers'])}")
                if details['shared_effects']:
                    shared_mechanics.append(f"Effects: {', '.join(details['shared_effects'])}")
                similarity_details = '\n'.join(shared_mechanics)

            # Insert card into treeview
            self.recommendations_tree.insert(
                "", tk.END,
                values=(
                    card['name'],
                    f"{card['edhrecRank']:.0f}",
                    f"{card['similarity_score']:.3f}",
                    f"{card['manaValue']}",
                    card.get('text', 'No card text available'),
                    card.get('keywords', 'No keywords'),
                    similarity_details
                )
            )

        # Display clusters
        for i, cluster in enumerate(results['clusters'], 1):
            cluster_node = self.clusters_tree.insert(
                "", tk.END,
                text=f"Cluster {i} (Avg Rank: {cluster['centroid_rank']:.0f})"
            )

            # Add cards to cluster with details
            for card in cluster['cards'][:10]:  # Top 10 cards per cluster
                self.clusters_tree.insert(
                    cluster_node, tk.END,
                    text=f"{card['name']} (Rank: {card['edhrecRank']:.0f}, MV: {card['manaValue']})"
                )

    def analyze_future_card(self):
        """Analyze future card"""
        # Get color identity
        color_identity = [color for color, var in self.color_vars.items() if var.get()]

        try:
            mana_value = float(self.mana_value_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid mana value.")
            return

        self.status_var.set("Analyzing future card...")

        def run_analysis():
            try:
                results = self.deck_builder.analyze_future_card(
                    card_name=self.future_name_entry.get(),
                    card_text=self.card_text_entry.get("1.0", tk.END).strip(),
                    mana_value=mana_value,
                    color_identity=color_identity
                )
                self.root.after(0, lambda: self.display_future_results(results))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.root.after(0, lambda: self.status_var.set("Ready"))

        threading.Thread(target=run_analysis, daemon=True).start()

    def display_future_results(self, results: Dict):
        """Display future card analysis results"""
        self.future_results.delete("1.0", tk.END)

        # Display predicted rank
        self.future_results.insert(tk.END,
                                   f"Predicted EDHRec Rank: {results['card_analysis']['predicted_rank']:.2f}\n\n")

        # Display mechanics
        self.future_results.insert(tk.END, "Mechanics Found:\n")
        for mech_type, mechs in results['card_analysis']['mechanics'].items():
            if mechs:
                self.future_results.insert(tk.END, f"{mech_type.capitalize()}: {mechs}\n")

        # Display top recommendations
        self.future_results.insert(tk.END, "\nTop Synergistic Cards:\n")
        for card in results['recommended_cards'][:5]:
            self.future_results.insert(tk.END, f"- {card['name']}\n")

    def run(self):
        """Start the GUI"""
        self.root.mainloop()


if __name__ == "__main__":
    app = MTGDeckBuilderGUI()
    app.run()