```mermaid
graph TD
    %% 스타일 정의
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef model fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef view fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef controller fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef output fill:#ffe0b2,stroke:#e65100,stroke-width:2px;

    %% 1. 입력 레이어
    subgraph Input_Layer [Layer 1: Input Sources]
        Webcam(📷 Webcam / OpenCV Frame)
        KeyboardListner(⌨️ Global Hotkey Listener)
    end

    %% 2. 메인 애플리케이션 (PyQt6)
    subgraph Main_App [Layer 2: Main Application (MVC Pattern)]
        
        %% Controller (지휘자)
        subgraph Controller_Group [Controller (Brain & Logic Flow)]
            MainCtrl(Main Controller)
            InputHandler(Input Handler)
            ActionDispatch(Action Dispatcher)
        en

        %% Model (데이터 & 판단)
        subgraph Model_Group [Model (Core Logic & State)]
            GestureEngine(🖐️ Gesture Recognition Engine<br>MediaPipe / AI Model)
            SafetyLogic(🛡️ Safety System<br>Deadman Switch Check)
            ModeState(🔄 Mode Manager<br>State: PPT/Media/Game)
        end

        %% View (화면)
        subgraph View_Group [View (GUI & Feedback)]
            MainWindow(🖥️ Main Window<br>Camera Feed Display)
            Overlay(🎨 Overlay / HUD<br>Stataus Bar & Icons)
            TrayIcon(🔽 System Tray Icon)
        end
    end

    %% 3. 출력 레이어
    subgraph Output_Layer [Layer 3: System Outputs]
        PPT_Ctrl(📊 PPT Control<br>Keyboard Simulation)
        Media_Ctrl(▶️ Media Control<br>YouTube/Volume)
        Game_Ctrl(🏎️ Game Interface<br>Virtual Joystick)
    end

    %% 데이터 흐름 연결 (Flow)
    Webcam -->|Raw Frame| InputHandler
    KeyboardListner -->|Wake Up Signal| MainCtrl

    InputHandler -->|Frame Data| MainCtrl
    MainCtrl -->|Process Request| GestureEngine
    
    GestureEngine -->|Landmarks Data| SafetyLogic
    SafetyLogic -->|Safety Status: Lock/Unlock| ModeState
    
    ModeState -->|Current Mode Info| ActionDispatch
    SafetyLogic -->|Safety Check Passed?| ActionDispatch

    MainCtrl -->|Update UI Signal| MainWindow
    ModeState -->|State Change Signal| Overlay
    
    ActionDispatch -->|Command| PPT_Ctrl
    ActionDispatch -->|Command| Media_Ctrl
    ActionDispatch -->|Command| Game_Ctrl

    %% 스타일 적용
    class Webcam,KeyboardListner input;
    class GestureEngine,SafetyLogic,ModeState model;a`
    class MainWindow,Overlay,TrayIcon view;
    class MainCtrl,InputHandler,ActionDispatch controller;
    class PPT_Ctrl,Media_Ctrl,Game_Ctrl output;a
```