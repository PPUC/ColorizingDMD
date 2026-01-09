#include <QApplication>
#include <QGuiApplication>
#include <QIcon>

#include "MainWindow.h"

int main(int argc, char *argv[])
{
    QGuiApplication::setAttribute(Qt::AA_DontShowIconsInMenus, false);
    QApplication app(argc, argv);

    MainWindow window;
    window.setWindowIcon(QIcon(":/app/app.ico"));
    window.show();

    return app.exec();
}
