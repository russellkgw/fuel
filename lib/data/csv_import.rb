class Data::CsvImport
  require 'csv'

  def initialize
  end

  def import(csv_file, expected_structure)
    file_data = parse_file(csv_file)

    if valid_file?(expected_structure, file_data)
      { valid: true, data: file_data }
    else
      # TODO: report errors
      { valid: false, data: ['An error has occurred during the parsing of the files, please re evaluate before uploading again.'] }
    end
  end

  def parse_file(csv_file)
    CSV.read(csv_file)
  end

  def valid_file?(expected_structure, data)
    data.present? && header_count_valid?(expected_structure.keys, data.first) &&
        header_order_valid?(expected_structure.keys, data.first) && col_data_valid?(data, expected_structure.values)
  end

  def header_count_valid?(expected, actual)
    expected.count == actual.count
  end

  def header_order_valid?(expected, actual)
    (0..(expected.count - 1)).each { |i| (return false) if (expected[i] != actual[i]) }
  end

  def col_data_valid?(data, valid_cols)
    data.delete_at(0)

    data.each do |row|
      row.each_with_index do |col, index|
        if valid_cols[index][:permitted].any?
          return false unless valid_cols[index][:permitted].include?(col)
        end
      end
    end

    true
  end

end