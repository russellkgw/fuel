class Data::ExchangeRateImport < Data::CsvImport

  STRUCTURE = { 'Base' => { type: 'String', permitted: ['USD'] },
                'Currency' => { type: 'String', permitted: ['ZAR'] },
                'Value' => { type: 'Decimal', permitted: [] },
                'Date' => { type: 'Date', permitted: [] },
                'Source' => { type: 'String', permitted: ['https://openexchangerates.org/'] },
                'Update' => { type: 'String', permitted: ['yes', 'no'] } }

  VALID_HEADERS = STRUCTURE.keys

  def self.import(file)
    exchange_import = new
    result = exchange_import.import(file, STRUCTURE)

    if result[:valid] == true
      exchange_import.insert_data(result[:data])
      result[:valid]
    else
      result[:valid]
    end
  end

  def initialize
    super
  end

  def insert_data(data)
    data.each do |item|
      record = ExchangeRate.where(base: item[0], currency: item[1], date: Date.parse(item[3])).first

      if record.present? && item[5] == 'yes'
        record.update({ base: item[0],
                        currency: item[1],
                        rate: item[2],
                        date: Date.parse(item[3]),
                        source: item[4] })
      elsif record.blank?
        ExchangeRate.create({ base: item[0],
                              currency: item[1],
                              rate: item[2],
                              date: Date.parse(item[3]),
                              source: item[4] })
      end
    end
  end
end