import QtQuick 2.7
import QtQuick.Controls 2.0
import QtQuick.Layouts 1.3
import QtMultimedia 5.5

import com.qubicaamf.vision 1.0

ApplicationWindow {
    visible: true
    width: 800
    height: 600
    title: qsTr("Markdetector Example")

    Camera {
        id: camera
    }

    MarkerDetectorFilter {
        id: markerDetectorFilter

        onMarkerFound: {
            markerID.text = id;
        }
    }

    Timer {
        id: timer
        interval: 2000
        running: false
        repeat: false

        onTriggered: {
            calibrationFilter.showNegative = false;
        }
    }

    VideoOutput {
        id: videoOutput
        source: camera

        anchors.fill: parent

        focus : visible // to receive focus and capture key events when visible
        filters: [markerDetectorFilter]
    }

    Rectangle {
        anchors.right: parent.right
        anchors.top: parent.top

        color: "black"

        width: 300
        height: 20

        opacity: 0.5

        Label {
            id: markerID
            anchors.fill: parent
            color: "white"
            text: ""
        }
    }
}
