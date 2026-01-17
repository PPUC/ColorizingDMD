#include <QApplication>
#include <QGuiApplication>
#include <QIcon>
#include <QCoreApplication>

#include "MainWindow.h"

int main(int argc, char *argv[])
{
    QGuiApplication::setAttribute(Qt::AA_DontShowIconsInMenus, false);
    QApplication app(argc, argv);
    QCoreApplication::setOrganizationName("PPUC");
    QCoreApplication::setApplicationName("PPUC-Serum-Colorizer");
    QCoreApplication::setApplicationVersion("0.1.0");

    MainWindow window;
    window.setWindowIcon(QIcon(":/app/app.png"));
    window.show();

    return app.exec();
}
