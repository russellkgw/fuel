class Data::OilPriceImport < Data::CsvImport
  STRUCTURE = { 'Currency' => { type: 'String', permitted: ['USD'] },
                'Date' => { type: 'String', permitted: [] },
                'Value' => { type: 'Date', permitted: [] },
                'Source' => { type: 'String', permitted: ['https://www.quandl.com/api/v3/datasets/EIA/PET_RBRTE_D'] },
                'Update' => { type: 'String', permitted: ['yes', 'no'] } }

  VALID_HEADERS = STRUCTURE.keys

  def self.import(file)
    oil_import = new
    result, data = oil_import.import(file, STRUCTURE)

    if result
      oil_import.insert_data(data)
      result
    else
      result
    end
  end

  def initialize
    super
  end

  def insert_data(data)
    data.each do |item|
      record = OilPrice.where(currency: item[0], date: Date.parse(item[1])).first

      if record.present? && item[4] == 'yes'
        record.update({ price: item[2] })
      elsif record.blank?
        OilPrice.create({ currency: item[0],
                          date: Date.parse(item[1]),
                          price: item[2],
                          source: item[3] })
      end
    end
  end
end