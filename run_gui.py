from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from qt_material import build_stylesheet
import sys
sys.path.append('./core/')
from miniML_gui import minimlGuiMain


app = QApplication(sys.argv)
app.setWindowIcon(QIcon('minML_icon.png'))
main = minimlGuiMain()
extra = {'density_scale': '-1',}
app.setStyleSheet(build_stylesheet(theme='light_blue.xml', invert_secondary=False, 
                                    extra=extra, template='miniml.css.template'))
main.show()
sys.exit(app.exec_())