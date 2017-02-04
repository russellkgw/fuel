class Data::Handler

  EXCHANGE_RATE_FILE = 'exchange_rate_file'
  OIL_PRICE_FILE = 'oil_price_file'
  FUEL_PRICE_FILE = 'fuel_price_file'

  def self.process_csv_files(data_files)
    new.process_csv_files(data_files)
  end

  def initialize
  end

  def process_csv_files(data_files)
    process_result = []
    data_files.each do |key, val|
      if val.present?
        process_result << "#{key} upload success: #{handle_file(key, val)}"
      end
    end

    process_result.join(' - ')
  end

  def handle_file(file_type, file)
    case file_type
      when EXCHANGE_RATE_FILE
        Data::ExchangeRateImport.import(file)
      when OIL_PRICE_FILE
        Data::OilPriceImport.import(file)
      when FUEL_PRICE_FILE
        Data::FuelPriceImport.import(file)
      else
        'file not supported.'
    end
  end
end