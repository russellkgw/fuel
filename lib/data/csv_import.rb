class Data::CsvImport
  require 'csv'

  def initialize
  end

  def import(csv_file, expected_headers)
    file_data = parse_file(csv_file)

    if valid_file?(expected_headers, file_data)
      { valid: true, data: file_data }
    else
      { valid: false, data: ['An error has occurred during the parsing of the files, please re evaluate before uploading again.'] }
    end
  end

  def parse_file(csv_file)
    CSV.read(csv_file)
  end

  def valid_file?(expected_headers, data)
    data.present? && header_count_valid?(expected_headers, data.first) && header_order_valid?(expected_headers, data.first)
  end

  def header_count_valid?(expected, actual)
    expected.count == actual.count
  end

  def header_order_valid?(expected, actual)
    (0..(expected.count - 1)).each { |i| (return false) if (expected[i] != actual[i]) }
  end
end