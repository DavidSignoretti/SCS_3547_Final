The Hypthesis
How does the Emergency Medical Service position an ambulance to a given geographic region?
The province is divided into different geographic response areas called "Response Areas." These Response Areas are from a few city blocks to hundreds of square KM. The project was simply a proof of concept that could Deep Q Learning predict the next call's location. Even though geographic boundaries define the response areas, a linear array can represent response areas. A linear numpy array probability distortion where the highest value in the array is the next response location.

It is merely an asymmetrical game with two players: an antagonist (the patients) and a sympathetic (the ambulance). Both players are allowed to change the state of the game without waiting for their turn. To keep the problem simple to start a single ambulance is used. The DQN tracks the location of the ambulance position using a zero-filled numpy array.

The hypothesis is --- maybe from the results of a simple model.

Method
The DQN Linear network predicts (kind of) the best location to place an ambulance based on the highest probability of a patient location.

The antagonist (the patient) hyperparameter ANTAGONISTIC_MOVE_RATE can change the rate the probability distribution changes the state. Encapsulated of the antagonist state is linear the numpy array antagonistic_state.

By moving the ambulance, the sympathetic changes the ambulance's position as part of the gameplay and the game rewards the sympathetic when in the correct location. The DQN is trying to position the ambulance in the correct location. sympathetic_state

        argmax(antagonistic_state) -> [0.02138027 0.01243725 0.32124701 ... ]
        argmax(sympathetic_state) ->  [0             0           1      ...]
        in this example the reward would be 50 because the index of antagonistic_state == sympathetic_state

The movment of the abulance is 0 to move right and 1 to move left and the function sympathetic_move handles the move. The function also check to make sure the move is with the array.

Results
First, although very noisy, the loss rate is decreasing, and the rate the ambulance is at the correct location is increasing. For the first few hundred iterations the loss value increases sharply. After a second spike around 500 iterations and then flattens to around 1500 iterations, the loss rate is continuously decreasing—the prediction of where the ambulance should follow the same trajectory. After approximately 1500 iterations, the predictions start becoming more accurate.

With more work and experimentation, a DQN could assist in determining where to locate an ambulance.
