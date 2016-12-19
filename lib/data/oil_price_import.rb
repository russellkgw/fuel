class Data::OilPriceImport < Data::CsvImport
  VALID_HEADERS = %W(Currency Value Date Source)

  def self.import(file)
    oil_import = new
    result = oil_import.import(file, VALID_HEADERS)

    if result[:valid] == true
      oil_import.insert_data(result[:data])
      result[:valid]
    else
      result[:valid]
    end
  end

  def initialize
    super
  end

  def create_update(data)
    existing_records = OilPrice.all

    # remove headers
    data.delete_at(0)

    data.each do |item|
      record = existing_records.where(base: item[0], currency: item[1], date: Date.parse(item[3])).first

      if record.present?
        record.update({ base: item[0], currency: item[1], value: item[2], date: Date.parse(item[3]), source: item[4] })
      else
        ExchangeRate.create({ base: item[0], currency: item[1], value: item[2], date: Date.parse(item[3]), source: item[4] })
      end
    end
  end
end