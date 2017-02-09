class Ann::Neuron

  def initialize
    @alphabet = alphabet
    @training_iterations = 1000
    @theta = 1.0

    @weights = alphabet_weights
    @weight_mod_value = 0.001

    @number = 5

    @global_errors = []

  end

  # This trains network to recognise @number
  def train_network!
    # STEP 1: iterate over training, zero Error and Weight update values
    (1..@training_iterations).each do |iteration|

      max_activation_output = 0.0
      iteration_error = 0.0
      weight_update_values = zero_weight_update_values

      # For each letter in the alphabet
      @alphabet.each_with_index do |letter, letter_index|
        activation_output = @theta * @weights[0] # multiply theta by weight[0]...bias?

        # For each char in letter multiply char value by its corresponding weight, add to activation output
        letter.split('').each_with_index do |char, char_index|
          activation_output += char.to_d * @weights[char_index + 1] # Activation
        end

        # If is letter supplied then compute the iteration_error and weight update values
        if letter_index == @number
          iteration_error += (1.0 - activation_output) ** 2
          weight_update_values[0] += (1.0 - activation_output) * @theta

          letter.split('').each_with_index do |char, char_index|
            weight_update_values[char_index + 1] += (1.0 - activation_output) * char.to_d
          end
        else # not the letter we are looking for.
          iteration_error += (0.0 - activation_output) ** 2
          weight_update_values[0] += (0.0 - activation_output) * @theta

          letter.split('').each_with_index do |char, char_index|
            weight_update_values[char_index + 1] += (0.0 - activation_output) * char.to_d
          end
        end

        current_max_activation_output = [max_activation_output, activation_output].max
        if current_max_activation_output > max_activation_output
          max_activation_output = current_max_activation_output
        end

        puts "Training iteration: #{iteration}, Current max activation output: #{current_max_activation_output}, Corresponding letter: #{letter}" # add letter index ?

      end

      @global_errors << iteration_error

      # Update the weights for next iteration
      (0..25).each do |index|
        @weights[index] += @weight_mod_value * weight_update_values[index] # Update the weight
      end
    end

  end

  # Each item is a letter with 25 chars
  def alphabet
    alphabet = ['1111110001111111000110001']
    alphabet << '1111110001111111000111111'
    alphabet << '1111110000100001000011111'
    alphabet << '1110010010100011001011100'
    alphabet << '1111110000111111000011111'
    alphabet << '1111110000111111000010000'
    alphabet << '1111110000100001000111111'
    alphabet << '1000110001111111000110001'
    alphabet << '1111100100001000010011111'
    alphabet << '1111100100001000010011100'
    alphabet <<	'1000110010111001001010001'
    alphabet <<	'1000010000100001000011111'
    alphabet <<	'1000111011101011000110001'
    alphabet <<	'1000111001101011001110001'
    alphabet <<	'1111110001100011000111111'
    alphabet <<	'1111110001111111000010000'
    alphabet <<	'1111110001101011001111111'
    alphabet <<	'1111110001111111001010001'
    alphabet <<	'1111110000111110000111111'
    alphabet <<	'1111100100001000010000100'
    alphabet <<	'1000110001100011000111111'
    alphabet <<	'1000110001010100101000100'
    alphabet <<	'1010110101101011010111111'
    alphabet <<	'1000101010001000101010001'
    alphabet <<	'1000110001010100010000100'
    alphabet <<	'1111100010001000100011111'
  end

  def alphabet_weights
    weights = []
    26.times do weights << 1.0 / (Random.new.rand(1..100)) end

    weights
  end

  # Values used to update the weights
  def zero_weight_update_values
    weight_update_values = []
    26.times do weight_update_values << 0.0 end

    weight_update_values
  end
  
end