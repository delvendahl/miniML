import numpy as np
from PyQt5.QtWidgets import QSplitter


class SplitterLayoutController:
    """Encapsulates splitter visibility toggles and size cache management."""

    def __init__(self, *, splitter2: QSplitter, splitter3: QSplitter):
        """
        Store splitter references and initialize pane-size caches.
        """
        self.splitter2 = splitter2
        self.splitter3 = splitter3

        self._store_size = self.splitter2.sizes()
        self._store_size_a = self._store_size[0]
        self._store_size_b = self._store_size[2]
        self._store_size_c = self.splitter3.sizes()

    def toggle_table(self) -> None:
        """
        Toggle visibility of the table pane in the outer splitter.
        """
        if 0 in self.splitter3.sizes():
            self.splitter3.setSizes(self._store_size_c)
            return

        self._store_size_c = self.splitter3.sizes()
        self.splitter3.setSizes([np.sum(self.splitter3.sizes()), 0])

    def toggle_plot(self) -> None:
        """
        Toggle visibility of the lower analysis plot pane.
        """
        sizes = self.splitter2.sizes()
        if sizes[2] == 0:  # panel is hidden
            sizes[0] = 0 if sizes[0] == 0 else self._store_size[0]
            sizes[1] = (
                (np.sum(sizes[0:3]) - self._store_size_b)
                if sizes[0] == 0
                else (self._store_size[1] - self._store_size_b)
            )
            sizes[2] = self._store_size_b
            self._store_size = sizes
        else:  # panel is shown
            self._store_size = sizes
            self._store_size_b = sizes[2]
            sizes[0] = 0 if sizes[0] == 0 else sizes[0]
            sizes[1] = np.sum(sizes[0:3]) if sizes[0] == 0 else np.sum(sizes[1:3])
            sizes[2] = 0
        self.splitter2.setSizes(sizes)

    def toggle_prediction(self) -> None:
        """
        Toggle visibility of the upper prediction pane.
        """
        sizes = self.splitter2.sizes()
        if sizes[0] == 0:  # panel is hidden
            sizes[0] = self._store_size_a
            sizes[1] = (
                (np.sum(sizes[0:3]) - self._store_size_a)
                if sizes[2] == 0
                else (self._store_size[1] - self._store_size_a)
            )
            sizes[2] = 0 if sizes[2] == 0 else self._store_size[2]
            self._store_size = sizes
        else:  # panel is shown
            self._store_size = sizes
            self._store_size_a = sizes[0]
            sizes[1] = np.sum(sizes[0:3]) if sizes[2] == 0 else np.sum(sizes[0:2])
            sizes[2] = 0 if sizes[2] == 0 else sizes[2]
            sizes[0] = 0
        self.splitter2.setSizes(sizes)
